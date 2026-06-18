from fucrimodo.analysis.multi_run_analysis import (
    MultiRunData,
    get_multi_run_overview,
    get_all_global_statistics_overview
)

def main(
    multi_run_dir: str,
    verbose: bool = True,
    row: int | None = None,
    show: bool = True,
    save_dir: str | None = None,
):
    multi_run_data = MultiRunData(multi_run_dir)

    print("________________________________________________________")
    print("Multi Run Overview:")
    overview = get_multi_run_overview(multi_run_data)
    print(overview)
    print()

    print("________________________________________________________")
    print("Global Statistics Overview:")
    global_stats_overview = get_all_global_statistics_overview(multi_run_data)
    print(global_stats_overview)

