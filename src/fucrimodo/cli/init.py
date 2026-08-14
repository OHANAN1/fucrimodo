import os
from pathlib import Path
import click


class Runner:
    def __init__(self, save_dir: Path, verbose: bool, skip_user_confirm: bool = False):
        self.save_dir = save_dir.resolve()
        self.verbose = verbose
        self.skip_user_confirm = skip_user_confirm

    def run(self):
        """Copy the fucrimodo_lab template to the desired location."""

        # Check if dir already exists
        fucrimodo_lab_dir = self.save_dir / "fucrimodo_lab"
        if os.path.isdir(fucrimodo_lab_dir):
            raise click.ClickException(
                f"The directory {fucrimodo_lab_dir} already exists."
            )

        # User has to confirm that dir is created
        if not self.skip_user_confirm:
            if not click.confirm(f"Create fucrimodo lab in {self.save_dir}?"):
                raise click.ClickException("Aborted.")

        from importlib.resources import files

        lab_template_path = files("fucrimodo") / "lab_template"
        self._copy_tree(lab_template_path, fucrimodo_lab_dir)

        click.echo(f"Success!")
        click.echo(f"Lab created in {fucrimodo_lab_dir}.")
        click.echo()
        click.echo("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        click.echo("                      Fucrimodo Lab                         ")
        click.echo("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        click.echo("Configure, run, store, and analyse reproducible multi-stage")
        click.echo("     optimization experiments with ease and fishes.        ")
        click.echo("        Quick! Go to the lab: `cd fucrimodo_lab.`          ")
        click.echo("    For more information, please check the README.md.      ")
        click.echo()
        click.echo(r"""
                                           .
                Max   /\            .     .
                    _/./            .     .
                 ,-'    `-:..-'/     .     .
                : o )      _  (     .     .
                "`-....,--; `-.\    .    .
                    `'             .    /mlb""")
        click.echo(" Brew a potion that reverts the descriptors to atomic form!")

    def _copy_tree(self, source, dest: Path) -> None:
        """Recursively copy a Traversable package-data tree to the filesystem."""
        dest.mkdir(parents=True, exist_ok=True)

        for item in source.iterdir():
            dest_item = dest / item.name
            if item.is_dir():
                self._copy_tree(item, dest_item)
            else:
                # read_bytes/write_bytes works whether the package is installed
                # as a regular directory or inside a zip/wheel
                dest_item.write_bytes(item.read_bytes())


@click.command(help="Generate a fucrimodo_lab directory with default configs.")
@click.option(
    "-s",
    "--save_dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path.cwd(),
    help="Path to where the fucrimodo_lab directory should be created. Defaults to current dir.",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Create the directory without asking for confirmation.",
)
@click.option("-v", "--verbose", is_flag=True, help="More output.")
def cli(save_dir, yes, verbose):
    runner = Runner(save_dir=save_dir, verbose=verbose, skip_user_confirm=yes)
    runner.run()
