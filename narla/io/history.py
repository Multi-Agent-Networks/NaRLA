import os
import narla


def save_history_as_data_frame(name: str, history: narla.history.History):
    """
    Save the History object as a Pandas DataFrame. Will be saved to ``<narla_trial_path/name.csv>``

    :param name: Name of DataFrame
    :param history: History object
    """
    if narla.settings.results_directory:
        data_frame = history.to_data_frame()

        file = os.path.join(narla.io.format_trial_path(narla.settings), name + ".csv")
        data_frame.to_csv(file)
