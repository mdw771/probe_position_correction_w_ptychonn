import json
import glob
import os.path


class JSONModifier:

    def __init__(self, mod_dict, new_name_appendix=''):
        self.mod_dict = mod_dict
        self.new_name_appendix = new_name_appendix

    def run(self):
        flist = glob.glob('*.json')
        for f in flist:
            d = json.load(open(f, 'r'))
            for k, v in self.mod_dict.items():
                d[k] = v
            json.dump(d, open(os.path.splitext(f)[0] + self.new_name_appendix + os.path.splitext(f)[1], 'w'),
                      sort_keys=False, indent=2, separators=(',', ': '))


if __name__ == '__main__':
    mod_dict = {
        "use_baseline_offsets_for_unregistered_points": True
    }
    modifier = JSONModifier(mod_dict, new_name_appendix='_decimation')
    modifier.run()
