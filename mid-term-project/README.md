# Machine Learning Zoomcamp Mid-Term Project
The mid-term project tackles the prediction of resale flats in Singapore.

Singapore has one of the highest home-ownership (?) in the world where majority
of the homeowners are public houses managed by the Housing and Development
Board (HDB).<super>[1][wiki]</super> Despite these being public houses, owners can sell
their housing units in the resale market, subject to certain restrictions.
As public housing makes up the majority of the transactions in Singapore's
property market, this project tackles the prediction of resale prices
based on the data the Singapore Government publishes.<super>[2][data]</super>

This project contains the following artifacts:

* `resale-flat-price-2017-2021.csv`: contains the data downloaded from
  Singapore Government data website.
* `requirements.txt`: states the Python packages necessary for running
  this mid-term project.
* `notebook.ipynb`: contains Exploratory Data Analysis (EDA), feature
  transformation, multiple model training, model selection, and model
  hyperparameter-tuning.
* `train.py`: contains the functions to create the pipeline of the selected
  and tuned model from `notebooks.ipynb`.
* `train_script.py`: runs `train.py` and exports the trained model as
  `pipeline.bin`; having this separate file overcomes the issue of pickling
  `FunctionTransformer` in `__main__`.<super>[3][issue]</super>
* `predict.py`: runs a web-service for the resale price prediction.
* `sample-data.json`: contains a single sample extracted from the original
  data source for testing the web-service.
* `pipeline.bin.zip`: zips a pre-trained model for ease of use.

## Testing the Web Service

To test the web service, either run `python train_script.py` to train and export
the model, or unzip `pipeline.bin.zip` first. Then run `python predict.py`.
It'll start the web service, listening at `0.0.0.0:8080`.

Then run the following command to send the data in `sample-data.json`
to interact with the service:

```bash
curl -X POST -H "Content-Type: application/json" --data "@./sample-data.json" http://localhost:8080/
```

[wiki]: https://en.wikipedia.org/wiki/Public_housing_in_Singapore
[data]: https://data.gov.sg/dataset/resale-flat-prices
[issue]: https://github.com/scikit-learn/scikit-learn/issues/12904
