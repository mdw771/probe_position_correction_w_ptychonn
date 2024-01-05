import json
import glob


class JSONModifier:

    def __init__(self, mod_dict):
        self.mod_dict = mod_dict

    def run(self):
        flist = glob.glob('*.json')
        for f in flist:
            d = json.load(open(f, 'r'))
            for k, v in self.mod_dict.items():
                d[k] = v
            json.dump(d, open(f, 'w'), sort_keys=False, indent=2, separators=(',', ': '))


if __name__ == '__main__':
    mod_dict = {
        "registration_method_multiiter": ["hybrid", "hybrid", "hybrid"],
        "num_neighbors_collective": 4,
        "smooth_constraint_weight_multiiter": [0, 0, 0],
        "hybrid_registration_algs_multiiter": [["error_map_expandable", "sift"], ["error_map_expandable", "sift"], ["error_map_expandable", "sift"]],
        "hybrid_registration_tols_multiiter": [[0.3, 0.15], [0.3, 0.15], [0.3, 0.15]]
    }
    modifier = JSONModifier(mod_dict)
    modifier.run()
