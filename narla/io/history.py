import os
import narla


def save_history_as_data_frame(name: str, history: narla.history.History):
    if narla.settings.results_directory:
        data_frame = history.to_data_frame()

        file = os.path.join(narla.io.format_trial_path(narla.settings), name + ".csv")
        data_frame.to_csv(file)
