import click

from fucrimodo.analysis.multi_run_analysis import (
    MultiRunData,
    get_all_global_statistics_overview,
    get_multi_run_overview,
)


def main(
    dir_path: str,
    verbose: bool = True,
    row: int | None = None,
    show: bool = True,
    save_dir: str | None = None,
):
    multi_run_data = MultiRunData(dir_path)

    click.echo("________________________________________________________")
    click.echo("Multi Run Overview:")
    overview = get_multi_run_overview(multi_run_data)
    click.echo(overview)
    click.echo()

    click.echo("________________________________________________________")
    click.echo("Global Statistics Overview:")
    global_stats_overview = get_all_global_statistics_overview(multi_run_data)
    click.echo(global_stats_overview)
