# Inspired by DEAP libarRunning y

from deap import tools, base, creator
import random
from tqdm import tqdm
from fucrimodo.core.utils.custom_soap import CustomSOAP
import warnings
from icecream import ic
import numpy as np
import ase

def get_stream_str(
    record: dict, 
    text_colors: list[str] = ["\033[30m", "\033[30m"], # black, black
    bg_colors: list[str] = ["\033[43m", "\033[46m"], # yellow, cyan
) -> str:
    RESET = "\033[0m"

    max_key_len = int(max([len(key) for key in record.keys()]))
    stream_str = ""
    i = 0
    for key, value in record.items():

        if i % 2 == 0:
            stream_str += f"{text_colors[0]}{bg_colors[0]}"
        else:
            stream_str += f"{text_colors[1]}{bg_colors[1]}"

        stream_str += f"{key:<{max_key_len}}: "
        for key2, value2 in value.items():
            if key2 != "gen" and key2 != "nevals":
                stream_str += f"{key2}={value2:9.9f}, "

        stream_str += f"{RESET}\n"
        i += 1

    return stream_str


def reset_individual(individual) -> None:
    if hasattr(individual, "fitness"):
        if hasattr(individual.fitness, "values"):
            del individual.fitness.values
    if hasattr(individual, "soap_features"):
        del individual.soap_features


def create_offspring(
    population: list,
    toolbox: base.Toolbox,
    cxpb: float,
    mutpb: float,
) -> tuple[list[ase.Atoms], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    r"""Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.
              And a dict with the crossover data and a dict with the mutation data.
              This data contains the number of times the crossover or mutation
              was called and how many times it was successful.
              The keys are the names of the crossover or mutation operators
              and the values are dicts with the keys "called" and "successful".

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [base.deepcopy(ind) for ind in population]

    crossover_data = {}
    mutation_data = {}

    # Apply crossover and mutation on the offspring
    iter_range = range(1, len(offspring), 2)
    performed_crossover = np.zeros(len(offspring))
    with tqdm(
        iter_range, desc=" --> Performing Crossover", leave=False, ascii=True
    ) as pbar:
        for i in iter_range:
            if random.random() < cxpb:
                offspring[i - 1], offspring[i], success_bool, crossover_name = toolbox.mate( # type: ignore
                    offspring[i - 1],
                    offspring[i]
                )

                if success_bool:
                    performed_crossover[i - 1] = 1
                    performed_crossover[i] = 1

                if crossover_name in crossover_data.keys():
                    crossover_data[crossover_name]["called"] += 1
                else:
                    crossover_data[crossover_name] = {
                        "called": 1,
                        "successful": 0
                    }
                if success_bool:
                    crossover_data[crossover_name]["successful"] += 1

                reset_individual(offspring[i])
                reset_individual(offspring[i - 1])

            pbar.update(1)

    iter_range = range(len(offspring))
    performed_mutation = np.zeros(len(offspring))
    with tqdm(
        iter_range, desc=" --> Performing Mutation", leave=False, ascii=True
    ) as pbar:
        for i in iter_range:
            if random.random() < mutpb:
                offspring[i], success_bool, mutation_name = toolbox.mutate( # type: ignore
                    offspring[i]
                )
                if success_bool:
                    performed_mutation[i] = 1

                if mutation_name in mutation_data.keys():
                    mutation_data[mutation_name]["called"] += 1
                else:
                    mutation_data[mutation_name] = {
                        "called": 1,
                        "successful": 0
                    }

                if success_bool:
                    mutation_data[mutation_name]["successful"] += 1

                reset_individual(offspring[i])

            pbar.update(1)

    ic("Performed crossover on {} individuals".format(np.sum(performed_crossover)))
    ic("Performed mutation on {} individuals".format(np.sum(performed_mutation)))

    modified_offspring = []
    for i in range(len(offspring)):
        if performed_crossover[i] == 1 or performed_mutation[i] == 1:
            modified_offspring.append(offspring[i])

    ic("Modified {} individuals".format(len(modified_offspring)))
    ic("Only using modified individuals as offspring")

    return modified_offspring, crossover_data, mutation_data


def update_fitnesses(
    population: list,
    toolbox: base.Toolbox,
    soap_obj: CustomSOAP | None,
) -> int:
    """
    Checks which individuals in the population have invalid fitnesses.
    That means that the fitnesses have not been evaluated yet.
    Then evaluates the fitnesses of the invalid individuals.
    Returns the number of individuals that had their fitnesses updated.
    """

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    for ind in invalid_ind:
        try:
            ind.wrap()
        except Exception as e:
            warnings.warn(
                f"Error in wrapping individual. {e}"
            )

    fitnesses = []
    iter_range = range(len(invalid_ind))
    with tqdm(
        iter_range, desc=" --> Calculating Fitnesses", leave=False, ascii=True
    ) as pbar_inner:

        for i in iter_range:
            # Features are calculated in the first call to the fitness function
            # so multiple fitness evaluations do not calculate the features again
            if hasattr(invalid_ind[i], "soap_feature_vector"):
                del invalid_ind[i].soap_feature_vector

            fitnesses.append(
                toolbox.evaluate( # type: ignore
                    invalid_ind[i]
                ) 
            )
            pbar_inner.update(1)


    ic("Updating fitnesses")
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    return len(invalid_ind)


def setup_logbooks(
    fitness_stats: tools.MultiStatistics,
    global_stats: None | tools.MultiStatistics,
) -> tuple[tools.Logbook, None | tools.Logbook]:
    """
    Creates the logbooks for the fitness and global statistics.

    The fitness_logbook has the fields of the fitness_stats and the gen and 
    nevals fields.
    The global_logbook has the fields of the global_stats and the gen field.
    If global_stats is None, the global_logbook is also None.
    """

    fitness_logbook = tools.Logbook()
    fitness_logbook.header = ['gen', 'nevals'] + fitness_stats.fields # type: ignore # noqa

    if global_stats is not None:
        global_logbook = tools.Logbook()
        global_logbook.header = ['gen'] + global_stats.fields  # type: ignore # noqa
    else:
        global_logbook = None

    return fitness_logbook, global_logbook


def record_statistics(
    population: list[ase.Atoms],
    nevals: int,
    gen: int,
    fitness_logbook: tools.Logbook,
    fitness_stats: tools.MultiStatistics,
    global_logbook: None | tools.Logbook = None,
    global_stats: None | tools.MultiStatistics = None,
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict[str, dict[str, int]]] | None]:
    """
    Calculates the statistics of the population for the fitness and global stats.

    :returns: A tuple with the fitness_record and the global_record.
    """
    fitness_record = fitness_stats.compile(population)
    fitness_logbook.record(gen=gen, nevals=nevals, **fitness_record)

    if global_stats is not None and global_logbook is not None:
        global_record = global_stats.compile(population)
        global_logbook.record(gen=gen, **global_record)
    else:
        global_record = None

    return fitness_record, global_record
    
def record_modification_log(
    modification_log: dict[str, dict[str, list[int]]],
    modification_data: dict[str, dict[str, int]],
) -> None:
    """
    Records the modification data in the modification_log.
    """
    for modification_name in modification_data.keys():
        if modification_name in modification_log.keys():
            data = modification_data[modification_name]
            modification_log[modification_name]["called"].append(
                data["called"]
            )
            modification_log[modification_name]["successful"].append(
                data["successful"]
            )
        else:
            warnings.warn(
                f"Modification name {modification_name} not in modification log."
            )

def myEaSimple(
    population: list,
    toolbox: base.Toolbox,
    cxpb: float,
    mutpb: float,
    ngen: int,
    mutation_log: dict[str, dict[str, list[int]]],
    crossover_log: dict[str, dict[str, list[int]]],
    fitness_stats: tools.MultiStatistics,
    global_stats: None | tools.MultiStatistics = None,
    soap_obj: CustomSOAP | None = None,
    halloffame: tools.HallOfFame | None = None,
    verbose=__debug__,
    progress_bar_title: str = "Evolving...",
) -> tuple[list, tools.Logbook, None | tools.Logbook, dict[str, dict[str, list[int]]], dict[str, dict[str, list[int]]]]:
    """
    My own version of the DEAP eaSimple function.
    It uses break_condition, which is a function that takes the population
    as an argument and returns True if the evolution should be stopped.
    Also it has a survivor selection.
    And if set it calculates the soap descriptors for the individuals and
    stores them in the individual's attribute "soap_features".

    The toolbox should have the following methods:
    - evaluate(individual) -> tuple
    - select_parents(population, n_parents) -> list
    - select_survivors(population, n_survivors) -> list
    - mate(ind1, ind2) -> tuple
    - mutate(individual) -> tuple
    - break_condition(population) -> bool
    """
    # Delete possible fitness or soap_features attributes of the individuals
    for ind in population:
        reset_individual(ind)

    fitness_logbook, global_logbook = setup_logbooks(
        fitness_stats=fitness_stats,
        global_stats=global_stats,
    )
    nevals = update_fitnesses(population, toolbox, soap_obj)

    if halloffame is not None:
        halloffame.update(population)

    fitness_record, global_record = record_statistics(
        population=population,
        nevals=nevals,
        gen=0,
        fitness_logbook=fitness_logbook,
        fitness_stats=fitness_stats,
        global_logbook=global_logbook,
        global_stats=global_stats,
    )

    if verbose:
        print()
        print("Evolving...")
        print()
        print(f"Gen: 0, n_evals: {nevals}, pop_size: {len(population)}")
        stream_str = get_stream_str(fitness_record)

        if global_record is not None:
            stream_str += get_stream_str(
                global_record, 
                text_colors=["\033[30m", "\033[30m"], 
                bg_colors=["\033[42m", "\033[45m"]
            )

        print(stream_str)

    iter_range = range(1, ngen + 1)
    with tqdm(iter_range, desc=progress_bar_title) as outer_pbar:

        for gen in iter_range:

            ic("Evolving Gen: ", gen)
            ic("Population size: ", len(population))

            parents = toolbox.select_parents(  # type: ignore
                population, len(population)
            )

            ic("Selected {} parents".format(len(parents)))

            offspring, cross_data, mut_data = create_offspring(parents, toolbox, cxpb, mutpb)

            record_modification_log(crossover_log, cross_data)
            record_modification_log(mutation_log, mut_data)

            nevals = update_fitnesses(offspring, toolbox, soap_obj)

            population_pool = []
            population_pool += population
            for ind in offspring:
                if ind not in population_pool:
                    population_pool.append(ind)

            ic("Created population pool")
            ic(f"Population size: {len(population)}")
            ic(f"Population pool size: {len(population_pool)}")

            if len(population_pool) < len(population):
                warnings.warn(
                    "The population pool is smaller than the population size."
                    f"Population pool size: {len(population_pool)}, "
                    f"Population size: {len(population)}"
                    f"Gen: {gen}",
                )
                population_pool = population_pool + population

            new_population = toolbox.select_survivors(  # type: ignore
                population_pool, len(population)
            )

            ic("Selected {} survivors".format(len(new_population)))

            if halloffame is not None:
                halloffame.update(new_population)

            fitness_record, global_record = record_statistics(
                population=population,
                nevals=nevals,
                gen=0,
                fitness_logbook=fitness_logbook,
                fitness_stats=fitness_stats,
                global_logbook=global_logbook,
                global_stats=global_stats,
            )
            population[:] = new_population

            if verbose:
                outer_pbar.write(
                    f"Gen: {gen}, n_evals: {nevals}, pop_size: {len(population)}"
                )

                stream_str = get_stream_str(fitness_record)
                if global_record is not None:
                    stream_str += get_stream_str(
                        global_record, 
                        text_colors=["\033[30m", "\033[30m"], 
                        bg_colors=["\033[42m", "\033[45m"]
                    )

                outer_pbar.write(stream_str)
                outer_pbar.refresh()
                outer_pbar.update(1)

            if toolbox.break_condition(population, gen):  # type: ignore
                # outer_pbar.close()
                ic("Finished evolution")
                print()
                print("Break condition was met. Stopping evolution.")
                print()
                break

    return population, fitness_logbook, global_logbook, crossover_log, mutation_log
