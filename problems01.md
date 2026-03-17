# remaining problems

1) It seems like the priorities is not working properly. I have specified a number of priorities (stations) in the system_config.yaml but the script is still running other stations. 
2) the events are not generated in the way I want to be generated. I want the following to happen: 
    a) events are found and stored in a csv file (under event_data on the NAS). from event_detector.py
    b) event clips are generated using extract_event_clips.
    c) the folders under ../../../../../../mnt/BSP_NAS2_work/auklab_model/event_data/ has the correct structure  