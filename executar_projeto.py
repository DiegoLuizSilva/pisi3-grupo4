"""Executa 03–06; --in-process dispensa sockets do kernel Jupyter."""
from pathlib import Path
import os, sys
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MPLBACKEND','Agg')
import nbformat
ROOT=Path(__file__).resolve().parent

def in_process(nb):
    import io
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from IPython.display import display, Image
    import matplotlib.pyplot as plt
    shell=InteractiveShell.instance()
    shell.reset(new_session=False)
    def show(*args,**kwargs):
        for number in plt.get_fignums():
            buffer=io.BytesIO()
            plt.figure(number).savefig(buffer,format='png',dpi=110,bbox_inches='tight')
            display(Image(data=buffer.getvalue()))
        plt.close('all')
    plt.show=show
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code': continue
        count+=1
        print('  célula',count,flush=True)
        with capture_output() as cap:
            result=shell.run_cell(cell.source,store_history=False)
        cell.execution_count=count
        cell.outputs=[]
        for name,content in [('stdout',cap.stdout),('stderr',cap.stderr)]:
            if content: cell.outputs.append(nbformat.v4.new_output('stream',name=name,text=content))
        for out in cap.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(cap.stdout+cap.stderr) from (result.error_before_exec or result.error_in_exec)
        if cap.stdout: print(cap.stdout[-2500:],flush=True)

if __name__=='__main__':
    os.chdir(ROOT/'notebooks')
    for path in sorted((ROOT/'notebooks').glob('0[3-6]_*.ipynb')):
        print('Executando',path.name,flush=True)
        nb=nbformat.read(path,as_version=4)
        try:
            if '--in-process' in sys.argv: in_process(nb)
            else:
                from nbclient import NotebookClient
                NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT/'notebooks')}}).execute()
        finally: nbformat.write(nb,path)
        print('Concluído',path.name,flush=True)
