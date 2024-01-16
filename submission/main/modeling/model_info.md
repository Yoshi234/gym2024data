There are several pretrained models contained within the `/pretrained` folder. 
Each of these models has a different purpose and predicts a separate feature
of the dataset.
* r1: r1 is trained on base features to predict the difficulty score (`D_Score`) for each competitor
* r2: r2 is trained on base features and `D_Score` to predict `E_Score` for each competitor
* r3: r3 is trained on base features, `D_Score`, and `E_Score` to predict `Penalty` score for each competitor
* r4: r4 is trained on base features, `D_Score`, `E_Score`, and `Penalty` to predict `Rank` 
* r5: r5 is trained on base features, `D_Score`, `E_Score`, and `Penalty` to predict `Score`
* r6: r6 is trained on base features (only) to predict `Score` - 

Preferred models: 
* the preferred models for two tasks are `r1 -> r2 -> r3 -> r4` for predicting rank and `r6` for predicting score
* use a sequential model with `r4` for predicting rank
* use a single hit model for predicting score from base features

The preferred models are to be used in the simulator constructed