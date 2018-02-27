# Temporarily move to $HOME directory
pushd .
cd $HOME

if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
    MINICONDA=Miniconda-latest-Linux-x86_64.sh
elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    MINICONDA=Miniconda-latest-MacOSX-x86_64.sh
fi

MINICONDA_MD5=$(curl -s http://repo.continuum.io/miniconda/ | grep -A3 $MINICONDA | sed -n '4p' | sed -n 's/ *<td>\(.*\)<\/td> */\1/p')
wget http://repo.continuum.io/miniconda/$MINICONDA
bash $MINICONDA -b -p miniconda

export PATH=$HOME/miniconda/bin:$PATH
conda install --yes conda-build jinja2 anaconda-client pip
conda config --add channels omnia

# Return to original directory
popd