# FA_COMPACT_SETUP

Setting up a Flash Attention environment is often difficult and time-consuming. It typically requires multiple rounds of trial and error, which can lead to excessive GPU usage and wasted compute.

FA_COMPACT_SETUP simplifies this process by automatically identifying compatible versions and configuring the environment efficiently.


## Features

- Automatically detects compatible dependency versions  
- Sets up the environment with minimal manual effort  
- Reduces trial-and-error during installation  
- Supports both local (isolated) and global setups  
- Helps minimize unnecessary GPU usage  

## Installation Options

Two installation approaches are available depending on your requirements.

### Local Installation (Recommended)

This is the fastest and safest method.

- Uses `python setup.py`  
- Creates an isolated environment using micromamba  
- Avoids system-level conflicts  

#### Steps

```bash
python setup.py
```
#### Advantages
Fully isolated environment
No dependency conflicts
Faster setup
Reproducible

### Global Installation (Shell Script)

This method installs dependencies globally on the system.

#### Steps
```bash
chmod +x setup.sh
source ./setup.sh
```
# Advantages
Useful for system-wide setups
No separate environment required
# Limitations
Slower setup
Risk of conflicts with existing packages
Less reproducible
