""" This module manages the execution of a Runner object (monitor, simalator...).
Reads and writes OS configs, records execution infos.

TODO: clean up, add comments and maybe doctest
"""

import time
import datetime
import os
import re
import imp
import sys
import shutil
from optparse import OptionParser
sys.path.append('./Sources/')
from multimode import Multimode
from PIL import Image
import scenario
import system_info
import tracer_courbes_health_mode_simulator as Courbe_HMS
import tracer_courbes_mode_belief_monitor as Courbe_Mode_Belief
import tracer_courbes_rul_monitor as Courbe_RUL
import tracer_courbes_mode_trajectories_monitor as Courbe_Mode_Trajectories
import diagnoser,prognoser



#import notify2
#notify2.init('PHM')
#import mail

parser = OptionParser(usage='usage: %prog RUNNER_FILE CONFIG_FILE_RUNNER MODEL_FILE CONFIG_FILE_MODEL SCENARIO_FILE')
(options, args) = parser.parse_args()
if len(args) != 5:
    parser.error(parser.usage)

runnerName = re.split('[/.]', args[0])[-2]
configrunnerName = re.split('[/.]', args[1])[-2]
modelName = re.split('[/.]', args[2])[-2]
configmodelName = re.split('[/.]', args[3])[-2]
scenarioName = re.split('[/.]', args[4])[-2]
modelRepo = args[4].split('/')[0]
sys.path.append(modelRepo)

filepath = os.getcwd()
sourcepath = filepath + os.path.sep + "Sources"

if os.path.isfile('tmp'):
    os.remove('tmp')
if os.path.isfile(sourcepath + os.path.sep + 'tmp'):
    os.remove(sourcepath + os.path.sep +'tmp')

runnerlib = imp.load_source(runnerName, filepath + os.path.sep + args[0])
configrunnerlib = imp.load_source(configrunnerName, filepath + os.path.sep + args[1])
configmodellib = imp.load_source(configmodelName,filepath + os.path.sep + args[3])
modellib = imp.load_source(modelName, filepath + os.path.sep + args[2])

scenario = scenario.Scenario(args[4])

model = modellib.model(modelName, configmodellib)
runner = runnerlib.Runner(model, configrunnerlib)

bos = time.time() # beginning of simulation
try:
    
    st = runner.run(scenario) # return simulation time
except:
    # save data
    import sys, traceback
    traceback.print_exc(file=sys.stdout)
    print('error during the run -> save data')
    st = time.time() - bos

eos = time.time() # end of simulation
eosName = time.strftime('%Y-%m-%d_%Hh%M', time.localtime(eos))
mainRepo = modelRepo + os.path.sep + "Results " + scenarioName.split('_wrapped')[0] + os.path.sep
repoSimu = mainRepo + runnerName + '_' + eosName + os.path.sep 
csvFileName = repoSimu + eosName + '.csv'
gzFileName = repoSimu + eosName + '.csv.gz'
figFileNameHMS = repoSimu + eosName + ' ' + runnerName + ' Health Mode Simulator'
figFileNameRUL = repoSimu + eosName + ' ' + runnerName + ' Remaining Useful Life'
figFileNameMT = repoSimu + eosName + ' ' + runnerName + ' Mode Trajectories'
figFileNameMB = repoSimu + eosName + ' ' + runnerName + ' Mode Belief'
infoFileName = repoSimu + eosName + '.info'

# create repo
if not os.path.exists(mainRepo):
    os.mkdir(mainRepo)

if not os.path.exists(repoSimu):
    os.mkdir(repoSimu)

# write csv
runner.write(csvFileName)

# write infos
with open(infoFileName, 'w') as f:
    import platform

    infos = []
    infos.append('machine name = ' + platform.node())
    infos.append('python version = ' + platform.python_version())
    infos.append('system = ' + platform.system() + ' ' + platform.release())
    infos.append('distribution = ' + ' '.join(platform.dist()))
    infos.append('distribution details = ' + platform.version())
    infos.append('architecture = ' + ' '.join(platform.architecture()))
    infos.append('machine = ' + platform.machine())
    infos.append('processor = ' + platform.processor())
    infos.append('cpu number = ' + str(system_info.cpu_count()))
    infos.append('physical cpu number = ' + str(system_info.cpu_count_physical()))
    infos.append('cpu type = ' + system_info.cpu_info())
    infos.append('ram = ' + system_info.ram())
    infos.append('runner = ' + runnerName)
    infos.append('model = ' + modelName)
    infos.append('model mode number = ' + str(len(model.modesTable)))
    sp = sum(1 for _ in model.symbolic_places())
    np = sum(1 for _ in model.numerical_places())
    hp = sum(1 for _ in model.hybrid_places())
    pl = sum(1 for _ in model.place())
    infos.append('model place number = ' + '{} ({}, {}, {})'.format(pl, sp, np, hp))
    tr = sum(1 for _ in model.transition())
    infos.append('model transition number = ' + str(tr))
    try:
        pldiag = sum(1 for _ in runner.diag.place())
        infos.append('diagnoser place number = ' + str(pldiag))
        trdiag = sum(1 for _ in runner.diag.transition())
        infos.append('diagnoser transition number = ' + str(trdiag))
        gain = - round((trdiag - tr) / tr,2)  * 100
        infos.append('diagnoser transition gain = ' + '{}%'.format(gain))
    except:
        pass
    try:
        plprog = sum(1 for _ in runner.prog.place())
        infos.append('prognoser place number = ' + str(plprog))
        trprog = sum(1 for _ in runner.prog.transition())
        infos.append('prognoser transition number = ' + str(trprog))
        gain = - round((trprog - tr) / tr ,2) * 100
        infos.append('prognoser transition gain = ' + '{}%'.format(gain))
    except:
        pass
    infos.append('scenario = ' + scenarioName)
    infos.append('beginning of simulation = ' + time.strftime('%c', time.localtime(bos)))
    infos.append('end of simulation = ' + time.strftime('%c', time.localtime(eos)))
    infos.append('simulation time = ' + str(datetime.timedelta(seconds=st)))
    infos.append('model configuration = ' + configmodelName)
    infos.append('runner configuration = ' + configrunnerName)
    infos.sort()
    f.write('\n'.join(infos))
    f.write('\n')
    infos = ['{} = {}'.format(a,b) for a,b in configmodellib.__dict__.items() if a.isupper()]
    infos.sort()
    f.write('\n'.join(infos))
    infos = ['{} = {}'.format(a,b) for a,b in configrunnerlib.__dict__.items() if a.isupper()]
    infos.sort()
    f.write('\n'.join(infos))
print('files written in {}'.format(repoSimu))
os.mkdir(repoSimu + 'Figures')
os.mkdir(repoSimu + 'Files .dot')



if (runnerName == 'simulator') :
    #If the runner is the simulator, draw the health mode.
    Courbe_HMS.draw_health_mode(csvFileName,figFileNameHMS)
    f = Image.open(figFileNameHMS + ".png").show()
    #Copy the .csv file into the model repository and rename it with the scenario name. It allows an easier call to the monitor
    shutil.copyfile(csvFileName ,modelRepo + os.path.sep + scenarioName.split('_wrapped')[0] + '.csv') 
    #Move the created figure into the Figure repository    
    os.rename(figFileNameHMS +".png",repoSimu + "Figures" + os.path.sep + eosName + ' ' + runnerName + ' Health Mode Simulator.png') 



elif (runnerName == 'monitor') :
    #If the runner is the monitor, draw the Remaining Useful Life, the Mode Trajectories and the mode Belief
    Courbe_RUL.draw_rul(gzFileName,figFileNameRUL)
    Courbe_Mode_Trajectories.draw_mode_trajectories(gzFileName,figFileNameMT)
    Courbe_Mode_Belief.draw_mode_belief(gzFileName,figFileNameMB)
    f = Image.open(figFileNameRUL + ".png").show()
    f = Image.open(figFileNameMT + ".png").show()
    f = Image.open(figFileNameMB + ".png").show()
    #Move the created figures into the Figures repertory
    os.rename(figFileNameRUL +".png",repoSimu + "Figures" + os.path.sep + eosName + ' ' + runnerName + ' Remaining Useful Life.png') 
    os.rename(figFileNameMT +".png",repoSimu + "Figures" + os.path.sep + eosName + ' ' + runnerName + ' Mode trajectories.png') 
    os.rename(figFileNameMB +".png",repoSimu + "Figures" + os.path.sep + eosName + ' ' + runnerName + ' Mode Belief.png') 


os.chdir(repoSimu)
#Draw the Petri Net model, its mode representation, the diagnoser and the prognoser
model.draw()
Multimode(model).draw()
diagnoser.Diagnoser(model).draw()
prognoser.Prognoser(model).draw()

#Move the created .png and .svg into the Figures repertory
os.rename(modelName + ".png","Figures/" + modelName + ".png") 
os.rename(modelName + "_multimode.png","Figures" + os.path.sep + modelName + "_multimode.png") 
os.rename(modelName + "_prognoser.png","Figures" + os.path.sep + modelName + "_prognoser.png") 
os.rename(modelName + "_diagnoser.png","Figures" + os.path.sep + modelName + "_diagnoser.png") 
os.remove(modelName + ".svg")
os.remove(modelName + "_multimode.svg")
os.remove(modelName + "_diagnoser.svg")
os.remove(modelName + "_prognoser.svg")
#Move the created .dot files into the Files .dot repertory
os.rename(modelName + ".dot","Files .dot/" + modelName + ".dot") 
os.rename(modelName + "_multimode.dot","Files .dot" + os.path.sep + modelName + "_multimode.dot") 
os.rename(modelName + "_prognoser.dot","Files .dot" + os.path.sep + modelName + "_prognoser.dot") 
os.rename(modelName + "_diagnoser.dot","Files .dot" + os.path.sep + modelName + "_diagnoser.dot") 


#os.rename(


    
    

#notify2.Notification('Simulation over', filename).show()
#mail.notify('Simulation over', t)
