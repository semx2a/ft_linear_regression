#! /bin/bash

echoerr() { echo "$@" 1>&2; }

if ! command -v python3 &>/dev/null;
then
	echoerr -e "Please install python3 to run this project\n"
	exit 1
fi
install() {
	python3 setup.py sdist bdist_wheel
	pip install ./dist/ft_linear_regression-0.0.1.tar.gz
}

uninstall() {
	echo "Uninstalling ft_linear_regression"
	pip uninstall -y ft_linear_regression
	echo "Removing build files"
	rm -rf ./dist ./build ./ft_linear_regression.egg-info
	echo "Removing installed files"
	rm -rf ~/my_env/lib/python3.10/site-packages/ft_linear_regression*
}

case "$1" in
	install)
		install
		;;
	uninstall)
		uninstall
		;;
	*)
		echoerr "Usage: $0 {install|uninstall}"
		exit 1
		;;
esac
