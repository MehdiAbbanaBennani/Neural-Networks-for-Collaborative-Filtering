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
	sed -i 's/::/\,/g' ml-10M100K/ratings.dat
	cut -d',' -f1-3 ml-10M100K/ratings.dat > ratings10M.csv
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
	sed -i 's/::/\,/g' ml-1m/ratings.dat
	cut -d',' -f1-3 ml-1m/ratings.dat > ratings1M.csv
	echo 'Done!'

	echo 'Moving ...'
	mv ratings1M.csv ../Databases
	echo 'Done!'

	echo 'Cleaning...'
	rm -r ml-1m/
	rm -r ml-1m.zip 
	echo 'Done!'

	echo 'Import completed!'

elif [ $1 = "0" ]
then
	echo 'Downloading files'
	wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
	echo 'Done'

	echo 'Extracting files'
	unzip ml-100k.zip
	echo 'Done!'

	echo 'Preprocessing ...'
	sed -i 's/\t/\,/g' ml-100k/u.data
	sort -t , -k 1,1n -k 2,2n ml-100k/u.data > ml-100k/u_sorted.data
	cut -d',' -f1-3 ml-100k/u_sorted.data > ratings100k.csv
	echo 'Done!'

	echo 'Moving ...'
	mv ratings100k.csv ../Databases
	echo 'Done!'

	echo 'Cleaning...'
	rm -r ml-100k/
	rm -r ml-100k.zip
	echo 'Done!'

	echo 'Import completed!'

else
	echo 'please enter 1 for 1M and 2 for 10M'
fi





