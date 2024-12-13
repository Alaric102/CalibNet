from torch.utils.data import Dataset
from pathlib import Path
from pykitti import odometry

class KittiOdometryDataset(Dataset):
    def __init__(self, config: str):
        super(KittiOdometryDataset, self).__init__()
        base_path = Path(config['base_path'])

        self._sequences = []
        self._sequence_intervals = []
        for sequence in config['sequences']:
            self._sequences.append(odometry(base_path, sequence=sequence))
            next_interval = len(self._sequences[-1]) if not self._sequence_intervals else self._sequence_intervals[-1] + len(self._sequences[-1])
            self._sequence_intervals.append(next_interval)

    def __len__(self):
        return self._sequence_intervals[-1]
    
    def __getitem__(self, index):
        for sequence_id in range(len(self._sequences)):
            if index < self._sequence_intervals[sequence_id]:
                sequence_index = index if not sequence_id else index - self._sequence_intervals[sequence_id-1]
                return {
                    "rgb" : self._sequences[sequence_id].get_cam2(sequence_index),
                    "cloud" : self._sequences[sequence_id].get_velo(sequence_index),
                    "extrinsic" : self._sequences[0].calib.T_cam2_velo,
                    "intrinsic" : self._sequences[0].calib.P_rect_20
                } 
        raise ValueError(f"Out of range {self.__len__()}. index = {index}")