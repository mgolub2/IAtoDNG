import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import toga
from toga import Group
from toga.style.pack import COLUMN, Pack

try:  # for nuitka/running as is
    from sinar_ia import read_sinar, create_ia_dng, thumb_correct
except ImportError:  # for briefcase
    from .sinar_ia import read_sinar, create_ia_dng, thumb_correct


async def process_file(file_path: Path, output_dir: Path, table_index):
    if output_dir is None:
        output_dir = Path(".").resolve()
    raw = await read_sinar(file_path)
    await create_ia_dng(raw, output_dir)
    return table_index


def import_files_process(file_list, dest_path_name):
    for f in file_list:
        shutil.copyfile(f, dest_path_name / f.name)
    return dest_path_name


class IaApp(toga.App):
    def text_state(self):
        if self.output_path is None and not bool(self.file_dict):
            return "Please select input and output directories."
        if self.output_path is None and bool(self.file_dict):
            return "Please select the output directory."
        if self.output_path is not None and not bool(self.file_dict):
            return "Please select the input directory."
        if self.output_path is not None and bool(self.file_dict):
            return "Ready to process images!"

    def startup(self):
        self.main_window = toga.MainWindow(title=self.name)
        self.file_dict = dict()
        self.output_path = None


        self.file_table = toga.Table(
            headings=["IA File", "Processed"],
            on_select=self.load_meta_and_thumb,
            data=[],
        )

        self.right_container = toga.Box(style=Pack(direction=COLUMN, padding=50))
        self.console = toga.Label(text=self.text_state())
        self.img_view = toga.ImageView(
            id="view1", image=toga.Image(path="resources/iatodng.jpg")
        )
        self.right_container.add(self.img_view)
        self.right_container.add(toga.Divider())
        scroll = toga.ScrollContainer(content=self.console)
        self.right_container.add(scroll)

        split = toga.SplitContainer()
        split.content = [(self.file_table, 1), (self.right_container, 4)]
        import_cmd = toga.Command(
            self.import_folder,
            text="Import",
            tooltip="Imports a folder (for example, a compact flash card).",
            shortcut=toga.Key.MOD_1 + "i",
            icon="icons/pretty.png",
            group=Group.FILE,
            order=0,
        )
        open_folder_cmd = toga.Command(
            self.open_folder,
            text="Input Folder",
            tooltip="Opens a folder of Sinar .IA images",
            shortcut=toga.Key.MOD_1 + "o",
            icon="resources/folder.png",
            group=Group.FILE,
            order=1,
        )
        set_output_cmd = toga.Command(
            self.set_output,
            text="Output Folder",
            tooltip="Select the output folder for your DNGs",
            shortcut=toga.Key.MOD_1 + "d",
            icon="resources/export.png",
            group=Group.FILE,
            order=2,
        )
        process_cmd = toga.Command(
            self.start_background_processing,
            text="Convert",
            tooltip="Converts .IA files listed to DNGs.",
            shortcut=toga.Key.MOD_1 + "p",
            icon="resources/start.png",
            group=Group.FILE,
            order=3,
        )

        # self.main_window.toolbar.add(import_cmd)
        self.main_window.toolbar.add(open_folder_cmd)
        self.main_window.toolbar.add(set_output_cmd)
        self.main_window.toolbar.add(process_cmd)

        self.main_window.content = split
        self.main_window.show()

    async def print_console(self, line, end="\n"):
        self.console.text = f"{line}{end}{self.console.text}"
        self.console.refresh()

    async def import_folder(self, sender):
        input_path_name = await self.main_window.select_folder_dialog(
            title="Choose a directory to import."
        )
        await self.print_console(f"Selected {input_path_name.name} for import...")
        dest_path_name = await self.main_window.select_folder_dialog(
            title="Choose a directory to copy too."
        )
        if input_path_name is not None and dest_path_name is not None:
            ia_files = list(Path(input_path_name).glob("*.IA"))
            wr_files = list(Path(input_path_name).glob("*.WR"))
            br_files = list(Path(input_path_name).glob("*.BR"))
            file_list = ia_files + wr_files + br_files
            await self.print_console(
                f"Copying {len(file_list)} files from {input_path_name.name} to {dest_path_name.name}..."
            )
            self.pool.apply_async(
                import_files_process,
                args=(file_list, dest_path_name),
                callback=self.find_ia_files,
            )

    async def open_folder(self, sender):
        path_name = await self.main_window.select_folder_dialog(
            title="Choose a directory containing sinar .IA raw files"
        )
        if path_name is not None:
            await self.find_ia_files(path_name)

    async def set_output(self, sender):
        path_name = await self.main_window.select_folder_dialog(
            title="Choose an output directory for DNGs."
        )
        if path_name is not None:
            self.output_path = Path(path_name).resolve()
            await self.print_console(f"Set {path_name.name} as output directory.")
            await self.print_console(self.text_state())

    def process_done(self, table_index):
        self.file_table.data[table_index].processed = True
        self.print_console(f"Processed {self.file_table.data[table_index].ia_file}.")

    def process_failed(self, e):
        self.print_console(f"{e}")

    async def process(self, thing):
        for i, file_path in enumerate(self.file_dict.values()):
            await process_file(file_path, self.output_path, i)
            await self.print_console(f"Processed {file_path.name}.")
            self.file_table.data[i].processed = True
        await self.print_console(f"Done processing {len(self.file_dict)} files.")

    async def start_background_processing(self, sender):
        self.add_background_task(self.process)
        await self.print_console(f"Processing... {len(self.file_dict)} files.")

    async def find_ia_files(self, directory):
        self.file_dict = {f.name: f for f in Path(directory).glob("*.IA")}
        self.file_table.data = [
            {"ia_file": f, "processed": False} for f in self.file_dict.keys()
        ]
        await self.print_console(f"Opened {directory.name}")
        await self.print_console(self.text_state())

    async def load_meta_and_thumb(self, sender, row):
        file = self.file_dict[row.ia_file]
        sinar_ia_raw = await read_sinar(file)
        buffer = NamedTemporaryFile()
        pil_img = thumb_correct(sinar_ia_raw)
        pil_img.save(buffer, format="png", compress_level=0)
        buffer.seek(0)
        self.img_view.image = toga.Image(buffer.name)
        buffer.close()


def main():
    return IaApp("IAtoDNG", "io.maxg.iatodng", icon="resources/logo.png")


if __name__ == "__main__":
    main().main_loop()
