source /usr/local/Ascend/cann/set_env.sh

pip install -r requirements.txt
pip install -r requirements_dev.txt

pytest
