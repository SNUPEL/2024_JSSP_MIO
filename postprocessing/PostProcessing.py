import pandas as pd
import os
from collections import OrderedDict, defaultdict


def machine_log(config):
    df = pd.read_csv(config.filename['log'])
    df = df.drop(df.columns[0], axis=1)
    # Filter 'Started' and 'Finished' events
    df_started = df[df['Event'] == 'Started'].drop(['Event', 'Process'], axis=1).reset_index(drop=True)
    df_finished = df[df['Event'] == 'Finished'].drop(['Event', 'Process'], axis=1).reset_index(drop=True)

    machine_start = []
    machine_finish = []
    for i in range(config.n_machine):
        machine_start.append(df_started[(df_started['Machine'] == 'M' + str(i))])
        machine_finish.append(df_finished[(df_finished['Machine'] == 'M' + str(i))])

        machine_start[i].reset_index(drop=True, inplace=True)
        machine_finish[i].reset_index(drop=True, inplace=True)
    data = []

    for i in range(config.n_machine):
        for j in range(len(machine_finish[i])):
            temp = {'Machine': i,
                    'Job': machine_start[i].loc[j, 'Part'],
                    'Start': int(machine_start[i].iloc[j, 0]),
                    'Finish': int(machine_finish[i].iloc[j, 0]),
                    'Delta': int(machine_finish[i].iloc[j, 0] - machine_start[i].iloc[j, 0])}
            data.append(temp)

    data = pd.DataFrame(data)
    data = data.sort_values(by=['Start'])
    data.reset_index(drop=True, inplace=True)
    if config.save_machinelog:
        data.to_csv(config.filename['machine'])
    return data
