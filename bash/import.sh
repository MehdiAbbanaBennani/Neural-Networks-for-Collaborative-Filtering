#!/bin/bash

if [ $1 = "2" ]
then
	echo 'Downloading files'
	wget http://files.grouplens.org/datasets/movielens/ml-10m.zip
	echo 'Done'

	echo 'Extracting files'
	unzip ml-10m.zip 
	echo 'Done!'

	echo 'Preprocessing ...'
	sed -i 's/::/\;/g' ml-10M100K/ratings.dat
	cut -d';' -f1-3 ml-10M100K/ratings.dat > ratings10M.csv
	echo 'Done!'

	echo 'Moving ...'
	mv ratings10M.csv ../Databases
	echo 'Done!'

	echo 'Cleaning...'
	rm -r ml-10M100K/
	rm -r ml-10m.zip
	echo 'Done!'

	echo 'Import completed!'

elif [ $1 = "1" ]
then
	echo 'Downloading files'
	wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
	echo 'Done'

	echo 'Extracting files'
	unzip ml-1m.zip 
	echo 'Done!'

	echo 'Preprocessing ...'
	sed -i 's/::/\;/g' ml-1m/ratings.dat
	cut -d';' -f1-3 ml-1m/ratings.dat > ratings1M.csv
	echo 'Done!'

	echo 'Moving ...'
	mv ratings1M.csv ../Databases
	echo 'Done!'

	echo 'Cleaning...'
	rm -r ml-1m/
	rm -r ml-1m.zip 
	echo 'Done!'

	echo 'Import completed!'

else
	echo 'please enter 1 for 1M and 2 for 10M'
fi





