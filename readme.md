# Environment Setup
Create a virtual environment named `env`
```
python3.8 -m venv env
```
Activate the virtual environment
```
source env/bin/activate
```
Install all dependencies specified in the `requirements.txt` file
```
pip install -r requirements.txt
```
To install the prophet module ensure that host machine has gcc/g++ compilers installed.
In addition make sure their version is recent (https://stackoverflow.com/questions/54150650/how-to-update-gcc-from-4-8-to-8-2-on-rhel7, https://github.com/facebook/prophet/issues/1057, https://stackoverflow.com/questions/36327805/how-to-install-gcc-5-3-with-yum-on-centos-7-2)

# Generate Streaming Synthetic Benchmark and Run Evaluation
```bash
cd scripts/
./streaming_synthetic_data_generation.sh
```