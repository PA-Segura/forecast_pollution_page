conda activate dash-env
gunicorn -c gunicorn_config.py dash_ozono_v0_1:server
tmux new-session -s pronoz
# Inside tmux pronoz:
python dash_ozono_v002.py
