# NEOPIC
This repository is for the Particle-In-Neural Operators (PINOP) scheme which is part of the Helmholtz AI project NEOPIC

## Installation
We can either use a package manager such as conda, mamba or micromamba for installing or just use python virtual environments.
The latter may be preferred in the supercomputer as we can then use the already available modules which are optimized for the 
hardware and network configurations. 

### Micromamba based installation

Anaconda with its default channels have licensing restrictions hence here we use micromamba and only conda-forge channel which is free and 
open-source. 

1) Download and extract the binary of micromamba in a directory of your choice
```
mkdir -p micromamba
curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ./micromamba/bin --strip-components=1 bin/micromamba
```
2) Edit your ~/.bashrc, ~/.zshrc, or ~/.profile, depending on your shell:
```
export PATH="<micromamba_directory>/bin:$PATH"
```
Then reload your shell (this step needs to be done whenever bashrc gets changed. So we will not repeat this step but implictly understood)
```
source ~/.bashrc
```
3) Micromamba needs to inject a shell hook to manage environments properly. Add this to your shell config file:
```
eval "$(micromamba shell hook --shell bash)"  # Replace with your shell if not bash
```
Then re-source your shell again.

4) Now in order to make sure we only download and install packages from conda-forge and not from the defaults channel. We will create a `.condarc` file 
and set the path for it in our bashrc. Create `.condarc` file in for example `<micromamba_directory>` with the following contents
```
channels:
  - conda-forge

channel_priority: strict
default_channels: []
custom_channels: {}
``` 
5) Let's add the following lines in `bashrc` to set the root directory of micromamba, config file and binary location
```
export MICROMAMBA_RC=<micromamba_directory>/.condarc
export MAMBA_EXE=<micromamba_directory>/bin/micromamba
export MAMBA_ROOT_PREFIX=<micromamba_directory>

```
and then re-source it.

6) Now we can create the environment with the yaml file as
```
micromamba create -f NEOPIC/DSE_FNO/cleaned_environment_high_level.yml --strict-channel-priority
```

7) Activate the environment using 
```
micromamba activate dse
```
8) Now we can run the train script to see if it works
```
python3 train.py
```
or in supercomputers interactive node
```
srun --ntasks=1 --gres=gpu:1 python3 train.py
```


