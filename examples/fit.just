data/data.csv :
	curl (location) > data/data.csv

fit : data/data/csv, model.json
	bayes fit -m prior.json -d data/data/csv -o posterior.json

summary.csv : posterior.json
	bayes summary -m posterior.json -o summary.csv


