#!/usr/bin/env python

import nest
import numpy as np
import pylab
import parameters as p

class SpikingNeuralNetwork():
	def __init__(self):
		nest.ResetKernel()
		nest.SetKernelStatus({"local_num_threads" : 1, "resolution" : p.time_resolution})
		self.spike_generators_l = nest.Create("poisson_generator", p.resolution[0]//2*p.resolution[1], params=p.poisson_params)
		self.spike_generators_r = nest.Create("poisson_generator", p.resolution[0]//2*p.resolution[1], params=p.poisson_params)
		self.neuron_l = nest.Create("iaf_psc_alpha", params=p.iaf_params)
		self.neuron_r = nest.Create("iaf_psc_alpha", params=p.iaf_params)
		self.spike_detector_l = nest.Create("spike_detector", params={"withtime": True})
		self.spike_detector_r = nest.Create("spike_detector", params={"withtime": True})
		self.multimeter_l = nest.Create("multimeter", params={"withtime":True, "record_from":["V_m"]})
		self.multimeter_r = nest.Create("multimeter", params={"withtime":True, "record_from":["V_m"]})
		weights_l = np.fliplr(p.weights_l.T).reshape(p.weights_l.size)
		weights_r = np.fliplr(p.weights_r.T).reshape(p.weights_r.size)
		for i in range(weights_l.size):
			syn_dict = {"model": "static_synapse", 
						"weight": weights_l[i]}
			nest.Connect([self.spike_generators_l[i]], self.neuron_l, syn_spec=syn_dict)
		for i in range(weights_r.size):
			syn_dict = {"model": "static_synapse", 
						"weight": weights_r[i]}
			nest.Connect([self.spike_generators_r[i]], self.neuron_r, syn_spec=syn_dict)
		nest.Connect(self.neuron_l, self.spike_detector_l)
		nest.Connect(self.neuron_r, self.spike_detector_r)
		nest.Connect(self.multimeter_l, self.neuron_l)
		nest.Connect(self.multimeter_r, self.neuron_r)

	def simulate(self, dvs_data):
		time = nest.GetKernelStatus("time")
		nest.SetStatus(self.spike_generators_l, {"origin": time})
		nest.SetStatus(self.spike_generators_r, {"origin": time})
		nest.SetStatus(self.spike_generators_l, {"stop": p.sim_time})
		nest.SetStatus(self.spike_generators_r, {"stop": p.sim_time})
		data_l = np.array(dvs_data[:dvs_data.shape[0]//2,:])
		data_l = data_l.reshape(data_l.size)
		data_r = np.array(dvs_data[dvs_data.shape[0]//2:,:])
		data_r = data_r.reshape(data_r.size)
		for i in range(data_l.size):
			rate = data_l[i]/p.max_spikes
			rate = np.clip(rate,0,1)*p.max_poisson_freq
			nest.SetStatus([self.spike_generators_l[i]], {"rate": rate})
		for i in range(data_r.size):
			rate = data_r[i]/p.max_spikes
			rate = np.clip(rate,0,1)*p.max_poisson_freq
			nest.SetStatus([self.spike_generators_r[i]], {"rate": rate})
		nest.Simulate(p.sim_time)
		n_l = nest.GetStatus(self.spike_detector_l,keys="events")[0]["times"]
		n_r = nest.GetStatus(self.spike_detector_r,keys="events")[0]["times"]
		nest.SetStatus(self.spike_detector_l, {"n_events": 0})
		nest.SetStatus(self.spike_detector_r, {"n_events": 0})
		return n_l, n_r