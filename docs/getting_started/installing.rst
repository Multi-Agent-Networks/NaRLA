Installing
==========

NaRLA is supported for Python3.8, you will need ``python3.8-venv``

.. code-block:: console

    sudo apt-get install python3.8 python3.8-venv


Install
-------

.. code-block:: console

    mkdir -p ~/projects
    mkdir -p ~/python_environments
    cd ~/python_environments

    # Create a virtual environment
    python3.8 -m venv --system-site-packages narla
    alias narla=~/python_environments/narla/bin/python3
    echo 'alias narla=~/python_environments/narla/bin/python3' >> ~/.bashrc

    # Download and install the NaRLA packages
    cd ~/projects
    git clone git@github.com:Multi-Agent-Networks/NaRLA.git
    narla -m pip install -e NaRLA
