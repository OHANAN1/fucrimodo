def let_user_select_statistics_key(possible_stat_keys: list[str]) -> str:
    """Prompts the user to select one of the possible keys

    :param possible_stat_keys: list of statistic keys that can be used to analyse the run(s).

    :returns: The selected statistics key.

    :raise AssertionError: If user input is not an integer 
        or if integer is to big.
    """

    print("_____________________________________________________")
    print("Please choose the statistics key you want to analyse.")
    print()
    for i, stat_key in enumerate(possible_stat_keys):
        print(f"\t{i}: {stat_key}")

    print()
    selected_index = input("Type one of the corresponding numbers on the left: ")
    assert type(selected_index) != int, "Please write an integer number"
    assert int(selected_index)+1 <= len(possible_stat_keys), "The number you selected is to big."

    statistics_key = possible_stat_keys[int(selected_index)]
    print(f" -> Selected Key: {statistics_key}")
    return statistics_key
