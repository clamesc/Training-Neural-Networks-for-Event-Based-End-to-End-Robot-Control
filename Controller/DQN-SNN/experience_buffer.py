#!/usr/bin/env python

import numpy as np

class ExperienceBuffer():
    def __init__(self, total_size, resolution, buffer_size):
        self.pre_s = np.zeros((total_size,resolution[0]*resolution[1]),dtype=int)
        self.action = np.zeros((total_size),dtype=int)
        self.reward = np.zeros((total_size),dtype=float)
        self.post_s = np.zeros((total_size,resolution[0]*resolution[1]),dtype=int)
        self.terminal = np.zeros((total_size),dtype=bool)
        self.buffer_size = buffer_size
        self.total_size = total_size
        self.resolution = resolution
        self.count = 0
    
    def add(self,experience):
        if not self.count < self.total_size:
            self.pre_s = np.append(self.pre_s, np.zeros((self.total_size,self.resolution[0]*self.resolution[1]),dtype=int), axis=0)
            self.action = np.append(self.action, np.zeros((self.total_size),dtype=int), axis=0)
            self.reward = np.append(self.reward, np.zeros((self.total_size),dtype=float), axis=0)
            self.post_s = np.append(self.post_s, np.zeros((self.total_size,self.resolution[0]*self.resolution[1]),dtype=int), axis=0)
            self.terminal = np.append(self.terminal, np.zeros((self.total_size),dtype=bool), axis=0)
        self.pre_s[self.count] = experience[0]
        self.action[self.count] = experience[1]
        self.reward[self.count] = experience[2]
        self.post_s[self.count] = experience[3]
        self.terminal[self.count] = experience[4]
        self.count += 1

    def sample(self,size):
        # Get Batch Index
        if self.count < self.buffer_size:
            idx = np.random.choice(self.count,size,replace=False)
        else:
            idx = self.count*np.ones(size,dtype=int) - np.random.choice(self.buffer_size,size,replace=False)
        batch = []
        batch.append(np.where(self.pre_s[idx]>0, 1, 0))
        batch.append(self.action[idx])
        batch.append(self.reward[idx])
        batch.append(np.where(self.post_s[idx]>0, 1, 0))
        batch.append(self.terminal[idx])
        return batch