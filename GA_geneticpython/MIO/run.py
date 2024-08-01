from GA_geneticpython.MIO.objects_spearman import *

from Data.Dataset.Dataset import Dataset as d_user
from Data.Adams.abz5.abz5 import Dataset as d5
from Data.Adams.abz6.abz6 import Dataset as d6
from Data.Adams.abz7.abz7 import Dataset as d7
from Data.Adams.abz8.abz8 import Dataset as d8
from Data.Adams.abz9.abz9 import Dataset as d9

# from PMX_MIOreplacement import PMXCrossover

from GA_geneticpython.MIO.Visualize_evolution import show_evolution

from Config.Run_Config import Run_Config
from geneticpython.models import PermutationIndividual
from geneticpython import Population
from geneticpython.core.operators import RouletteWheelSelection, RouletteWheelReplacement, SwapMutation
from geneticpython import GAEngine
import time

data5 = d5()
data6 = d6()
data7 = d7()
data8 = d8()
data9 = d9()
data10 = d_user('test_10015.txt')
data11 = d_user('test_2020.txt')
data12 = d_user('test_3030.txt')
data13 = d_user('test_4040.txt')
data14 = d_user('test_5020.txt')

seed = 42
n_generation = 100
n_population = 100
n_selection = 40
step_size = 1 / (n_population + n_selection * n_generation)
swap_pairs = 1
p_mutation = 0.8
p_crossover = 0.8


def run_ga_instance(dataset, adaptive_, replacement_, crossover_, path_=None):
    n_op = dataset.n_op
    n_machine = dataset.n_machine
    if adaptive_:
        from PMX_plain import PMXCrossover
        crossover = PMXCrossover(pc=p_crossover)
        crossover.set_n(n_op, n_machine)
        pmx_type = "adaptive"
    elif replacement_:
        from PMX_MIOreplacement import PMXCrossover
        crossover = PMXCrossover(pc=p_crossover)
        crossover.set_n(n_op, n_machine)
        pmx_type = crossover.type
    elif crossover_:
        from PMX_MIOcrossover import PMXCrossover
        crossover = PMXCrossover(pc=p_crossover)
        crossover.set_n(n_op, n_machine)
        pmx_type = crossover.type
    else:
        from PMX_plain import PMXCrossover
        crossover = PMXCrossover(pc=p_crossover)
        crossover.set_n(n_op, n_machine)

        pmx_type = crossover.type

    op_data = dataset.op_data
    config = Run_Config(dataset.n_job, dataset.n_machine, dataset.n_op,
                        False, False, False,
                        False, False, False)

    indv_temp = PermutationIndividual(length=dataset.n_op, start=0)
    population = Population(indv_temp, size=n_population)
    selection = RouletteWheelSelection()

    mutation = SwapMutation(pm=p_mutation, n_points=swap_pairs)
    replacement = RouletteWheelReplacement()

    engine = GAEngine(population, selection=selection,
                      selection_size=n_selection,
                      crossover=crossover,
                      mutation=mutation,
                      replacement=replacement)

    evol = []
    popul = []
    makespan = []
    mio_value = []

    # @engine.maximize_objective
    @engine.minimize_objective
    def fitness(indv):
        raw = indv.chromosome.genes.tolist()
        # ind = Individual(config=config, seq=raw, op_data=op_data)
        index = sorted(range(len(raw)), key=lambda k: raw[k], reverse=False)
        result_sequence = [0 for i in range(len(raw))]

        for i, idx in enumerate(index):
            result_sequence[idx] = i
        ind = Individual(config=config, seq=result_sequence, op_data=op_data)
        evol.append(ind)

        makespan.append(ind.makespan)
        mio_value.append(ind.score)

        if adaptive_:
            upper_bound_1 = makespan[0]
            upper_bound_2 = mio_value[0]

            # 1. original settings for KIIE Conference
            # ratio_1 = 60. * step_size * len(evol)
            # ratio_2 = 100. - ratio_1
            # makespan_score = ratio_1 * (ind.makespan / upper_bound_1)
            # input_score = ratio_2 * (ind.score / upper_bound_2)
            # return makespan_score + input_score

            # 2. Revised settings for KIIE Conference - Ver1 => GA engine 설정이 minimize object 였기때문에 이건 잘못입력한 것
            # ratio_1 = 100. * step_size * len(evol)
            # ratio_2 = 100. - ratio_1
            # makespan_score = ratio_1 * (ind.makespan / upper_bound_1)
            # mio_score = np.exp(-ind.score / upper_bound_2)
            # input_score = ratio_2 * mio_score
            # return makespan_score + input_score

            # 3. Revised settings for KIIE Conference - Ver2
            ratio_1 = 20 + 80 * step_size * len(evol)
            ratio_2 = 100. - ratio_1
            makespan_score = ind.makespan / upper_bound_1
            mio_score = ind.score / upper_bound_2
            total_score = ratio_1 * makespan_score + ratio_2 * mio_score
            # print('Ratio_1 :',ratio_1, '\t| Ratio_2 :',ratio_2)
            return total_score

        else:

            return ind.makespan

    history = engine.run(generations=n_generation)
    # ans = engine.get_best_indv()
    # print(ans)
    # idx = makespan.index(min(makespan))

    # print(min(makespan))
    # print('\n\n')
    # print('score for minimum makespan:', mio_value[idx])
    # print('Total Collected Individuals : ', len(makespan))
    # print('Number of replacement:', crossover.num_called)
    # print('P_mio:', crossover.p_mio)

    show_evolution(makespan, mio_value,
                   title=dataset.name + '_' + str(n_generation) + 'gen' + '_MIO(' + pmx_type + ')_' + str(
                       min(makespan)),
                   text="* Swap" + str(swap_pairs) + "pair\nP(mutation)=" + str(
                       p_mutation) + "\n" + "P(crossover)=" + str(p_crossover)
                        + "\n" + "Population=" + str(n_population)
                        + "\n" + "# of Selection=" + str(n_selection), path=path_)

    data = pd.DataFrame({'makespan': makespan, 'score': mio_value})

    # 파일명 설정
    filename = dataset.name + '_' + str(n_generation) + 'gen' + '_MIO(' + pmx_type + ')_' + str(
        min(makespan))

    return (data, filename, min(makespan))
    return min(makespan)


def custom_sort(num_list, n_machine):
    return sorted(num_list, key=lambda x: (x % n_machine, x // n_machine))


if __name__ == '__main__':

    path = '/GA_geneticpython/MIO/KIIE_Journal_NewFitness\\'
    # for data in [data6]:
    result_dict = dict()
    idx = 0
    mode = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)}
    mode_eng = {0: 'Plain', 1: 'Fitness', 2: 'Replacement', 3: 'Crossover'}

    # for data in [data5]:
    for data in [data5, data6, data7, data8, data9, data10, data11, data12, data13, data14]:
        # for data in [data5, data6, data7, data8, data9, data10, data11, data12, data13, data14]:
        for n_mode in [1]:
            for _ in range(5):
                print('Now Running... ', end='')
                print('Data:', data.name, 'Mode:', mode_eng[n_mode])
                key = mode[n_mode]
                start = time.time()
                result, filename, min_makespan = run_ga_instance(data, key[0], key[1], key[2], path_=path)
                # min_makespan = run_ga_instance(data, key[0], key[1], key[2])
                end = time.time()
                result.to_csv(path + filename + "_" + str(round(end - start, 3)) + '.csv', index=False)

                idx += 1
                result_dict[idx] = {'Data': data.name, 'Mode': mode_eng[n_mode], 'Makespan': min_makespan,
                                    'Time': round(end - start, 4)}
                print('Experiment ', idx, ',\tMakespan : ', min_makespan)

    # DataFrame을 CSV 파일로 저장
    result_df = pd.DataFrame(result_dict).transpose()
    result_df.to_csv('Result_240728_newexperiment2.csv')
