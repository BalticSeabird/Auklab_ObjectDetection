


# Stage1 
20260510 - Detection model works pretty bad for Rost 3-4, but should be OK with some more training. 


# Stage4
Stage4 seems to work and is integrated in the main_orchestrator.py

It saved data in station specific dbs, under data/events_db

However, it needs fine tuning for arrivals with fish, especially for whether a bird actually arrives or not. 

Retraining the detection model to get better data for fish is also a priority, as I think there are some false negatives for the fish detection right now 
