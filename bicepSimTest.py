#To run this script correctly it needs to be placed in the Mujoco_py folder and run using python3


import mujoco_py
import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mujoco_py import load_model_from_xml, MjSim, MjViewer



def updateXML(s1,s2):       #method that inserts the site postions into the XML file
    xmlEdited="""
    <?xml version="1.0" ?>
    <mujoco model="Humanoid">
        <compiler inertiafromgeom="true" angle="degree"/>

        <default>
            <joint limited="true" damping="1" armature="0"/>
            <geom condim="1" material="matgeom"/>
            <motor ctrlrange="-.4 .4" ctrllimited="true"/>
        </default>

        <option timestep="0.005" iterations="50" tolerance="1e-10" solver="Newton" jacobian="dense" cone="pyramidal"/>

        <size nconmax="50" njmax="200" nstack="10000"/>

        <visual>
            <map force="0.1" zfar="30"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <quality shadowsize="2048"/>
            <global offwidth="800" offheight="800"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/> 

            <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>  

            <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278" 
                rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>  

            <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>

            <material name="matgeom" texture="texgeom" texuniform="true" rgba="0.8 0.6 .4 1"/>
        </asset>

        <worldbody>
            <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="matplane" condim="3"/>

            <light directional="false" diffuse=".2 .2 .2" specular="0 0 0" pos="0 0 5" dir="0 0 -1" castshadow="false"/>
            <light mode="targetbodycom" target="torso" directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 4.0" dir="0 0 -1"/>       

            <body name="torso" pos="0 0 1.3">
                <freejoint name="root"/>
                <geom name="torso1" type="capsule" fromto="0 -.07 0 0 .07 0"  size="0.07"/>
                <site name="s5" pos="0 -0.09 0.1" size="0.02"/>
                <site name="s6" pos="0.08 0.08 0.1" size="0.02"/>
                <geom name="head" type="sphere" pos="0 0 .19" size=".09"/>
                <geom name="uwaist" type="capsule" fromto="-.01 -.06 -.12 -.01 .06 -.12" size="0.06"/>
                <body name="lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0" >
                    <geom name="lwaist" type="capsule" fromto="0 -.06 0 0 .06 0"  size="0.06" />
                    <joint name="abdomen_z" type="hinge" pos="0 0 0.065" axis="0 0 1" range="0 1" damping="5" stiffness="20" armature="0.02" />
                    <joint name="abdomen_y" type="hinge" pos="0 0 0.065" axis="0 1 0" range="0 1" damping="5" stiffness="10" armature="0.02" />
                    <body name="pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0" >
                        <joint name="abdomen_x" type="hinge" pos="0 0 0.1" axis="1 0 0" range="0 1" damping="5" stiffness="10" armature="0.02" />
                        <geom name="butt" type="capsule" fromto="-.02 -.07 0 -.02 .07 0"  size="0.09" />
                        <body name="right_thigh" pos="0 -0.1 -0.04" >
                            <joint name="right_hip_x" type="hinge" pos="0 0 0" axis="1 0 0" range="0 5"   damping="5" stiffness="10" armature="0.01" />
                            <joint name="right_hip_z" type="hinge" pos="0 0 0" axis="0 0 1" range="0 3"  damping="5" stiffness="10" armature="0.01" />
                            <joint name="right_hip_y" type="hinge" pos="0 0 0" axis="0 1 0" range="0 2" damping="5" stiffness="20" armature="0.01" />
                            <geom name="right_thigh1" type="capsule" fromto="0 0 0 0 0.01 -.34"  size="0.06" />
                            <body name="right_shin" pos="0 0.01 -0.403" >
                                <joint name="right_knee" type="hinge" pos="0 0 .02" axis="0 -1 0" range="-3 -2" stiffness="1" armature="0.0060" />
                                <geom name="right_shin1" type="capsule" fromto="0 0 0 0 0 -.3"   size="0.049" mass="50"/>
                                <body name="right_foot" pos="0 0 -.39" >
                                    <joint name="right_ankle_y" type="hinge" pos="0 0 0.08" axis="0 1 0"   range="0 2" stiffness="4" armature="0.0008" />
                                    <joint name="right_ankle_x" type="hinge" pos="0 0 0.04" axis="1 0 0.5" range="0 2" stiffness="1"  armature="0.0006" />
                                    <geom name="right_foot_cap1" type="capsule" fromto="-.07 -0.02 0 0.14 -0.04 0"  size="0.027" mass="100"/>
                                    <geom name="right_foot_cap2" type="capsule" fromto="-.07 0 0 0.14  0.02 0"  size="0.027" mass="100"/>
                                </body>
                            </body>
                        </body>
                        <body name="left_thigh" pos="0 0.1 -0.04" >
                            <joint name="left_hip_x" type="hinge" pos="0 0 0" axis="-1 0 0" range="0 5"  damping="5" stiffness="10" armature="0.01" />
                            <joint name="left_hip_z" type="hinge" pos="0 0 0" axis="0 0 -1" range="0 3" damping="5" stiffness="10" armature="0.01" />
                            <joint name="left_hip_y" type="hinge" pos="0 0 0" axis="0 1 0" range="0 2" damping="5" stiffness="20" armature="0.01" />
                            <geom name="left_thigh1" type="capsule" fromto="0 0 0 0 -0.01 -.34"  size="0.06" />
                            <body name="left_shin" pos="0 -0.01 -0.403" >
                                <joint name="left_knee" type="hinge" pos="0 0 .02" axis="0 -1 0" range="-3 -2" stiffness="1" armature="0.0060" />
                                <geom name="left_shin1" type="capsule" fromto="0 0 0 0 0 -.3"   size="0.049" mass="50"/>
                                <body name="left_foot" pos="0 0 -.39" >
                                    <joint name="left_ankle_y" type="hinge" pos="0 0 0.08" axis="0 1 0"   range="0 2"  stiffness="4" armature="0.0008" />
                                    <joint name="left_ankle_x" type="hinge" pos="0 0 0.04" axis="1 0 0.5" range="0 2"  stiffness="1"  armature="0.0006" />
                                    <geom name="left_foot_cap1" type="capsule" fromto="-.07 0.02 0 0.14 0.04 0"  size="0.027" mass="100"/>
                                    <geom name="left_foot_cap2" type="capsule" fromto="-.07 0 0 0.14  -0.02 0"  size="0.027" mass="100"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
                <body name="right_upper_arm" pos="0 -0.17 0.06" >
                    <joint name="right_shoulder1" type="hinge" pos="0 0 0" axis="2 1 1"  range="-85 60" stiffness="1" armature="0.0068" />
                    <joint name="right_shoulder2" type="hinge" pos="0 0 0" axis="0 -1 1" range="-85 60" stiffness="1"  armature="0.0051" />
                    <geom name="right_uarm1" type="capsule" fromto="0 0 0 .16 -.16 -.16"  size="0.04 0.16" />
                    <site name="s1" pos="{}" size="0.02" rgba="10 0 0 10"/>
                    <site name="s4" pos="0 0.009 -0.06" size="0.02"/>
                    <site name="s9" pos="0.06 0.06 0" size="0.02"/>
                    <site name="s10" pos="-0.005 -0.08 -0.06" size="0.02"/>

                    <body name="right_lower_arm" pos=".18 -.18 -.18" >
                        <joint name="right_elbow" type="hinge" pos="0 0 0" axis="0 -1 1" range="-90 50"  stiffness="0" armature="0.0028" />

                        <geom name="right_larm" type="capsule" fromto="0.01 0.01 0.01 .17 .17 .17"  size="0.031" />
                        <geom name="right_hand" type="sphere" pos=".18 .18 .18"  size="0.04"/>
                        <site name="s2" pos="{}" size="0.02"/>
                        <site name="s3" pos="0 -0.06 -0.006" size="0.02"/>

                        
                    </body>
                </body>
                <body name="left_upper_arm" pos="0 0.17 0.06" >
                    <joint name="left_shoulder1" type="hinge" pos="0 0 0" axis="2 -1 1" range="0 5" stiffness="1" armature="0.0068" />
                    <joint name="left_shoulder2" type="hinge" pos="0 0 0" axis="0 1 1" range="0 5"  stiffness="1" armature="0.0051" />
                    <geom name="left_uarm1" type="capsule" fromto="0 0 0 .16 .16 -.16"  size="0.04 0.16" />
                    <body name="left_lower_arm" pos=".18 .18 -.18" >
                        <joint name="left_elbow" type="hinge" pos="0 0 0" axis="0 -1 -1" range="0 5" stiffness="0" armature="0.0028" />
                        <geom name="left_larm" type="capsule" fromto="0.01 -0.01 0.01 .17 -.17 .17"  size="0.031" />
                        <geom name="left_hand" type="sphere" pos=".18 -.18 .18"  size="0.04"/>
                    </body>
                </body>
            </body>
        </worldbody>

        <tendon>
            <spatial name="Bicep" width="0.01">
                <site site="s1"/>
                <site site="s2"/>
            </spatial>
            <spatial name="Tricep" width="0.01">
                <site site="s4"/>
                <site site="s10"/>
                <site site="s3"/>
            </spatial>
            <spatial name="Lateral Deltoid" width="0.01">
                <site site="s5"/>
                <site site="s1"/>
            </spatial>
            <spatial name="Front Deltoid" width="0.01">
                <site site="s6"/>
                <site site="s9"/>
            </spatial>
        </tendon>

        <actuator>
            <muscle name="Bicep" tendon="Bicep"/>
            <muscle name="Tricep" tendon="Tricep"/>
            <muscle name="Lateral Deltoid" tendon="Lateral Deltoid" force="-1" scale="1000"/>
            <muscle name="Front Deltoid" tendon="Front Deltoid" force="-1" scale="1000"/>
        </actuator>
    </mujoco>
    """.format(s1,s2)

    return xmlEdited
##############################################################################################################################

#v1 = "0 0 0.05"   #position for site one(Need to make it a list and use that to iterate all values)
v2 = "0.006 0.06 0.07"   #position for site two

with open('Data.csv', mode='w') as dataFile:        #opens and writes to the CSV file
        dataWrite = csv.writer(dataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        dataWrite.writerow(["Site Pos","Step/Time", "Actuator Force"])

min = np.array([0,0,0.05])
max = np.array([0.003,0.05,0.08])

p = np.linspace(min,max,5)
plt.savefig("test.png")

for num,j in enumerate(p):

    t = -0.4     #initial actuator value for the bottom position of the bicep
    dist = np.sqrt(np.square(j[0]-min[0])+np.square(j[1]-min[1])+np.square(j[2]-min[2]))
    var1 = str(round(j[0],2)) + " " + str(round(j[1],2)) + " " + str(round(j[2],2))
    print(type(var1))
    print(var1)
    
    xml = updateXML(var1,v2)

    model = load_model_from_xml(xml)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)

    x=[]
    y=[]

    for i in range(1200): #runs the simulation

        sim.data.ctrl[0] = t
        t += 0.001
        if t>0.6:
            break
        sim.step()
        with open('Data.csv', mode='a') as dataFile:
            dataWrite = csv.writer(dataFile, delimiter=',')
            dataWrite.writerow([var1,str(i),str(sim.data.actuator_force[0])])
        x.append(int(i)/200)
        #x.append(abs(float(sim.data.get_joint_qpos("right_elbow"))))
        y.append(abs(float(sim.data.actuator_force[0])))
        viewer.render()
        print(sim.data.get_joint_qpos("right_elbow"))
        if os.getenv('TESTING') is not None:
            break
    
    var1 = str(round(dist * 1000,1)) + " mm"
    #plt.plot(x[220:],y[220:], label=var1 ,lw = 0.5)
    plt.plot(x,y,label = var1,lw=0.5)
    plt.title('MuJoCo Force-Time')

    plt.xlabel('Force(N)')
    plt.ylabel('Time(seconds)')
    plt.legend(loc='lower right', borderaxespad=0., fontsize = 'xx-small')
    plt.savefig("test.png")




