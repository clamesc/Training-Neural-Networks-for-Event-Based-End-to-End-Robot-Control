function setLeftMotorVelocity_cb(msg)
    -- Left motor speed subscriber callback
    simSetJointTargetVelocity(leftMotor,msg.data)
end

function setRightMotorVelocity_cb(msg)
    -- Right motor speed subscriber callback
    simSetJointTargetVelocity(rightMotor,msg.data)
end

function resetRobot_cb(msg)
    -- Reset robot subscriber callback
    if msg.data then -- Outer Lane
        allModelObjects=simGetObjectsInTree(robotHandle) -- get all objects in the model
        simSetThreadAutomaticSwitch(false)
        for i=1,#allModelObjects,1 do
            simResetDynamicObject(allModelObjects[i]) -- reset all objects in the model
        end
        simSetObjectPosition(robotHandle,-1,{3.0,0.25,0.13879})
        simSetObjectOrientation(robotHandle,-1,{0.0,0.0,0.0})
        simSetThreadAutomaticSwitch(true)
    else -- Inner Lane
        allModelObjects=simGetObjectsInTree(robotHandle) -- get all objects in the model
        simSetThreadAutomaticSwitch(false)
        for i=1,#allModelObjects,1 do
            simResetDynamicObject(allModelObjects[i]) -- reset all objects in the model
        end
        simSetObjectPosition(robotHandle,-1,{9.25,7.0,0.13879})
        simSetObjectOrientation(robotHandle,-1,{0.0,0.0,3*math.pi/2})
        simSetThreadAutomaticSwitch(true)
    end
end

function getTransformStamped(objHandle,name,relTo,relToName)
    t=simGetSystemTime()
    p=simGetObjectPosition(objHandle,relTo)
    o=simGetObjectQuaternion(objHandle,relTo)
    return {
        header={
            stamp=t,
            frame_id=relToName
        },
        child_frame_id=name,
        transform={
            translation={x=p[1],y=p[2],z=p[3]},
            rotation={x=o[1],y=o[2],z=o[3],w=o[4]}
        }
    }
end

if (sim_call_type==sim_childscriptcall_initialization) then
    robotHandle=simGetObjectAssociatedWithScript(sim_handle_self)

    leftMotor=simGetObjectHandle("Pioneer_p3dx_leftMotor") -- Handle of the left motor
    rightMotor=simGetObjectHandle("Pioneer_p3dx_rightMotor") -- Handle of the right motor
    
    -- Check if the required ROS plugin is there:
    moduleName=0
    moduleVersion=0
    index=0
    pluginNotFound=true
    while moduleName do
        moduleName,moduleVersion=simGetModuleName(index)
        if (moduleName=='RosInterface') then
            pluginNotFound=false
        end
        index=index+1
    end

    -- Prepare DVS handle
    cameraHandle=simGetObjectHandle('DVS128_sensor')
    angle=simGetScriptSimulationParameter(sim_handle_self,'cameraAngle')
    if (angle>100) then angle=100 end
    if (angle<34) then angle=34 end
    angle=angle*math.pi/180
    simSetObjectFloatParameter(cameraHandle,sim_visionfloatparam_perspective_angle,angle)
    showConsole=simGetScriptSimulationParameter(sim_handle_self,'showConsole')
    if (showConsole) then
        auxConsole=simAuxiliaryConsoleOpen("DVS128 output",500,4)
    end
    showCameraView=simGetScriptSimulationParameter(sim_handle_self,'showCameraView')
    if (showCameraView) then
        floatingView=simFloatingViewAdd(0.2,0.8,0.4,0.4,0)
        simAdjustView(floatingView,cameraHandle,64)
    end

    if (not pluginNotFound) then
                -- Prepare the sensor publisher and the motor speed subscribers:
        dvsPub=simExtRosInterface_advertise('/dvsData', 'std_msgs/Int8MultiArray')
        transformPub=simExtRosInterface_advertise('/transformData', 'geometry_msgs/Transform')
        leftMotorSub=simExtRosInterface_subscribe('/leftMotorSpeed','std_msgs/Float32','setLeftMotorVelocity_cb')
        rightMotorSub=simExtRosInterface_subscribe('/rightMotorSpeed','std_msgs/Float32','setRightMotorVelocity_cb')
        resetRobotSub=simExtRosInterface_subscribe('/resetRobot','std_msgs/Bool','resetRobot_cb')
    end
end

if (sim_call_type==sim_childscriptcall_cleanup) then
    if not pluginNotFound then
        -- Following not really needed in a simulation script (i.e. automatically shut down at simulation end):
        simExtRosInterface_shutdownPublisher(dvsPub)
        simExtRosInterface_shutdownPublisher(transformPub)
        simExtRosInterface_shutdownSubscriber(leftMotorSub)
        simExtRosInterface_shutdownSubscriber(rightMotorSub)
        simExtRosInterface_shutdownSubscriber(resetRobotSub)
    end
    if auxConsole then
        simAuxiliaryConsoleClose(auxConsole)
    end
end

if (sim_call_type==sim_childscriptcall_sensing) then 
    -- Read and formate DVS data at each simulation step
    if notFirstHere and not pluginNotFound then
        r,t0,t1=simReadVisionSensor(cameraHandle)
    
        if (t1) then
            ts=math.floor(simGetSimulationTime()*1000)
            newData={}
            for i=0,(#t1/3)-1,1 do
                newData[1+i*2]=math.floor(t1[3*i+2])
                newData[2+i*2]=math.floor(t1[3*i+3])
                --newData=newData..string.char(timeStampByte1)
                --newData=newData..string.char(timeStampByte2)

                if (showConsole) then
                    if (t1[3*i+1]>0) then
                        onOff=", on"
                    else
                        onOff=", off"
                    end
                    simAuxiliaryConsolePrint(auxConsole,"time="..ts.." ms, x="..math.floor(t1[3*i+2])..", y="..math.floor(t1[3*i+3])..onOff.."\n")
                end
            end
        end
        simExtRosInterface_publish(dvsPub,{data=newData})
        p=simGetObjectPosition(robotHandle,-1)
        o=simGetObjectQuaternion(robotHandle,-1)
        simExtRosInterface_publish(transformPub, {translation={x=p[1],y=p[2],z=p[3]},rotation={x=o[1],y=o[2],z=o[3],w=o[4]}})
    end
    notFirstHere=true
    
    -- newData now contains the same data as would the real sensor (i.e. for each pixel that changed:
    -- 7 bits for the x-coord, 1 bit for polatiry, 7 bits for the y-coord, 1 bit unused, and 1 word for the time stamp (in ms)
    -- You can access this data from outside via various mechanisms. For example:
    --
    -- 
    -- simSetStringSignal("dataFromThisTimeStep",newData)
    --
    -- Then in a different location:
    -- data=simGetStringSignal("dataFromThisTimeStep")
    --
    --
    -- Of course you can also send the data via tubes, wireless (simTubeOpen, etc., simSendData, etc.)
    --
    -- Also, if you you cannot read the data in each simulation
    -- step, then always append the data to an already existing signal data, e.g.
    --
    -- 
    -- existingData=simGetStringSignal("TheData")
    -- if existingData then
    --     data=existingData..data
    -- end
    -- simSetStringSignal("TheData",data)
end 
