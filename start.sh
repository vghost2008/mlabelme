#!/bin/bash
#export QT_DEBUG_PLUGINS=1
#export LD_LIBRARY_PATH=/home/wj/software/venv/lib/python3.8/site-packages/PyQt5/Qt5/lib
#source /home/wj/software/venv/bin/activate
#/mnt/data12/wj/mldata/eiseg/venv/bin/python labelme/__main__.py
export PYTHONPATH=${PYTHONPATH}:/home/wj/ai/work/wml
export QT_DEBUG_PLUGINS=1
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/PyQt5/Qt5/lib
/usr/bin/python3 labelme/__main__.py --labelflags '{.*: [ignore, difficult,crowd]}'
