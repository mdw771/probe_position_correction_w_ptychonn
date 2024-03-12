"""
Migrates the current repository to PtychoNN's `package` branch.
"""
import os
import shutil
import re
import argparse
import ast
import libcst as cst


class CodeMigrator:

    def __init__(self, source_dir, dest_dir):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.source_code_dir = os.path.join(self.source_dir, 'pppc')
        self.dest_code_dir = os.path.join(self.dest_dir, 'ptychonn', 'pospred')
        self.source_tester_dir = os.path.join(self.source_dir, 'tests')
        self.dest_tester_dir = os.path.join(self.dest_dir, 'tests')
        self.new_import_name = 'ptychonn.pospred'
        self.copied_files = {}

    def build_dir(self):
        dir_list = [self.dest_code_dir, self.dest_tester_dir]
        for d in dir_list:
            if not os.path.exists(d):
                os.makedirs(d)

    def copy_and_rename_files(self, fname_dict, default_dest_dir, root_source_dir):
        for source_fname in fname_dict.keys():
            dest_fname = fname_dict[source_fname]
            if dest_fname is None:
                dest_fname = os.path.join(default_dest_dir, os.path.relpath(source_fname, root_source_dir))
            if not os.path.exists(os.path.dirname(dest_fname)):
                os.makedirs(os.path.dirname(dest_fname))
            shutil.copy(source_fname, dest_fname)
            self.copied_files[source_fname] = dest_fname

    def copy_code(self):
        fname_dict = {
            os.path.join(self.source_code_dir, 'core.py'): None,
            os.path.join(self.source_code_dir, 'registrator.py'): None,
            os.path.join(self.source_code_dir, 'reconstructor.py'): None,
            os.path.join(self.source_code_dir, 'configs.py'): None,
            os.path.join(self.source_code_dir, 'util.py'): None,
            os.path.join(self.source_code_dir, 'io.py'): None,
            os.path.join(self.source_code_dir, 'position_list.py'): None,
            os.path.join(self.source_code_dir, 'helper.py'): None,
            # os.path.join(self.source_code_dir, 'message_logger.py'): None,
        }
        self.copy_and_rename_files(fname_dict, default_dest_dir=self.dest_code_dir, root_source_dir=self.source_code_dir)

    def copy_tester(self):
        fname_dict = {
            os.path.join(self.source_tester_dir, 'test_multiiter_pos_calculation.py'): None,
            os.path.join(self.source_tester_dir, 'data', 'pred_test235', 'pred_phase.tiff'): os.path.join(self.dest_tester_dir, 'data', 'pospred', 'pred_test235', 'pred_phase.tiff'),
            os.path.join(self.source_tester_dir, 'data', 'config_235.json'): os.path.join(self.dest_tester_dir, 'data', 'pospred', 'config_235.json'),
            os.path.join(self.source_tester_dir, 'data', 'config_235.toml'): os.path.join(self.dest_tester_dir, 'data', 'pospred', 'config_235.toml'),
            os.path.join(self.source_tester_dir, 'data_gold', 'calc_pos_235.csv'): os.path.join(self.dest_tester_dir, 'data_gold', 'pospred', 'calc_pos_235.csv'),
            os.path.join(self.source_dir, 'readme.md'): os.path.join(self.dest_tester_dir, 'pospred_readme.md'),
        }
        self.copy_and_rename_files(fname_dict, default_dest_dir=self.dest_tester_dir, root_source_dir=self.source_tester_dir)

    def change_import(self):
        for dest_f in self.copied_files.values():
            if not os.path.splitext(dest_f)[1] == '.py':
                continue
            f = open(dest_f, 'r')
            lines = f.readlines()
            new_lines = []
            for i, l in enumerate(lines):
                if ('import pppc' in l) or ('from pppc' in l) or ('pppc.' in l):
                    if ('ptychonn' in l):
                        continue
                    l = l.replace('pppc', self.new_import_name)
                new_lines.append(l)
            f.close()
            f = open(dest_f, 'w')
            f.writelines(new_lines)
            f.close()

    def change_logging(self):
        for dest_f in self.copied_files.values():
            if not os.path.splitext(dest_f)[1] == '.py':
                continue
            f = open(dest_f, 'r')
            lines = f.readlines()
            new_lines = []
            has_import_logging = False
            for i, l in enumerate(lines):
                if 'import logging' in l:
                    has_import_logging = True
                if 'logger.info' in l:
                    l = l.replace('logger.info', 'logging.debug')
                # if 'import logging' in l:
                #     continue
                new_lines.append(l)
            if not has_import_logging:
                new_lines = [u'import logging\n',] + new_lines
            # new_lines = [u'import logging\n', u'logging.getLogger(__name__).setLevel(logging.INFO)\n', u'\n'] + new_lines
            f.close()
            f = open(dest_f, 'w')
            f.writelines(new_lines)
            f.close()

    def remove_unused_components(self):
        for fname in self.copied_files.values():
            if not os.path.splitext(fname)[1] == '.py':
                continue
            # self.remove_unused_components_from_file(fname, imported_modules_to_delete=['{}.message_logger'.format(self.new_import_name)])
            if os.path.basename(fname) == 'reconstructor.py':
                self.remove_unused_components_from_file(fname,
                                                        classes_to_delete=['PyTorchReconstructor',
                                                                           'ONNXTensorRTReconstructor',
                                                                           'DatasetInferencer',
                                                                           'ProbePositionList',
                                                                           'TileStitcher'],
                                                        imported_modules_to_delete=[
                                                            '{}.ptychonn.model'.format(self.new_import_name),
                                                            '{}.position_list'.format(self.new_import_name),
                                                            '{}.helper'.format(self.new_import_name),
                                                            'tqdm',
                                                            'scipy'
                                                        ])
            elif os.path.basename(fname) == 'registrator.py':
                self.remove_unused_components_from_file(fname,
                                                        classes_to_delete=['PhaseCorrelationRegistrationAlgorithm'],
                                                        imported_modules_to_delete=[
                                                            '{}.skimage_registration.phase_cross_correlation'.format(self.new_import_name)
                                                        ])
                self.remove_large_commented_blocks(fname, threshold=5)
                self.remove_lines_with(fname, key_phrases=["PhaseCorrelationRegistrationAlgorithm"])
            elif os.path.basename(fname) == 'core.py':
                self.remove_unused_components_from_file(fname,
                                                        imported_classes_to_delete=['PyTorchReconstructor'])
            elif os.path.basename(fname) == 'test_multiiter_pos_calculation.py':
                self.replace_lines_with(fname, "os.path.join('data', '", "os.path.join('data', 'pospred', '")
                self.replace_lines_with(fname, "os.path.join('data_gold',", "os.path.join('data_gold', 'pospred', ")
            self.remove_lines_with(fname, ['import logger'])
            self.add_blank_lines_before_class_definition(fname)

    def remove_unused_components_from_file(self, fname, classes_to_delete=(), imported_classes_to_delete=(),
                                           imported_modules_to_delete=()):
        f = open(fname, 'r')
        code = f.read()
        code = self.delete_classes_from_code(code, classes_to_delete)
        code = self.delete_imported_classes_from_code(code, imported_classes_to_delete)
        code = self.delete_imported_modules_from_code(code, imported_modules_to_delete)
        f.close()
        f = open(fname, 'w')
        f.write(code)
        f.close()

    def delete_classes_from_code(self, code, names_to_remove):
        module = cst.parse_module(code)

        class RemoveMyFunction(cst.CSTTransformer):
            def leave_ClassDef(self, original_node, updated_node):
                if original_node.name.value in names_to_remove:
                    return cst.RemoveFromParent()
                return updated_node

        new_module = module.visit(RemoveMyFunction())
        code = new_module.code.strip()
        return code

    def delete_imported_modules_from_code(self, code, names_to_remove):
        lines = code.split('\n')
        new_lines = []
        for i, l in enumerate(lines):
            delete_line = False
            if 'import ' in l:
                for name in names_to_remove:
                    if name in l:
                        delete_line = True
                        break
            if not delete_line:
                new_lines.append(l)
        code = '\n'.join(new_lines)
        return code

    def delete_imported_modules_from_code_ast(self, code, names_to_remove):
        lines = code.split('\n')
        for i, l in enumerate(lines):
            if not 'import' in l:
                continue
            tree = ast.parse(l)
            for node in ast.walk(tree):
                imported_names = []
                if isinstance(node, ast.Import):
                    imported_names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported_names = [node.module]
                for imported_name in imported_names:
                    if imported_name in names_to_remove:
                        tree.body.remove(node)
            l = ast.unparse(tree)
            lines[i] = l
        new_code = '\n'.join(lines)
        return new_code

    def delete_imported_classes_from_code(self, code, names_to_remove):
        lines = code.split('\n')
        new_lines = []
        for i, l in enumerate(lines):
            res = re.search(r'from (.+) import (.+)', l)
            if res:
                from_module = res.groups()[0]
                imported_classes = res.groups()[1]
                imported_classes = [x.strip() for x in imported_classes.split(',')]
                for imported_class in imported_classes:
                    if imported_class in names_to_remove:
                        imported_classes.remove(imported_class)
                if len(imported_classes) > 0:
                    l = 'from {} import {}'.format(from_module, ', '.join(imported_classes))
                else:
                    continue
            new_lines.append(l)
        code = '\n'.join(new_lines)
        return code

    def delete_imported_classes_from_code_ast(self, code, names_to_remove):
        lines = code.split('\n')
        new_lines = []
        for i, l in enumerate(lines):
            if not 'import' in l:
                new_lines.append(l)
                continue
            tree = ast.parse(l)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported_names = [a.name for a in node.names]
                    for i_name, imported_name in enumerate(imported_names):
                        if imported_name in names_to_remove:
                            node.names.remove(node.names[i_name])
                    if len(node.names) == 0:
                        tree.body.remove(node)
            l = ast.unparse(tree)
            new_lines.append(l)
        new_code = '\n'.join(new_lines)
        return new_code

    def remove_lines_with(self, fname, key_phrases):
        f = open(fname, 'r')
        lines = f.readlines()
        new_lines = []
        for l in lines:
            delete_line = False
            for phrase in key_phrases:
                if phrase in l:
                    delete_line = True
                    break
            if not delete_line:
                new_lines.append(l)
        f.close()
        f = open(fname, 'w')
        f.writelines(new_lines)
        f.close()

    def replace_lines_with(self, fname, original, replace):
        f = open(fname, 'r')
        lines = f.readlines()
        new_lines = []
        for l in lines:
            l = l.replace(original, replace)
            new_lines.append(l)
        f.close()
        f = open(fname, 'w')
        f.writelines(new_lines)
        f.close()

    def remove_large_commented_blocks(self, fname, threshold=5):
        f = open(fname, 'r')
        lines = f.readlines()
        new_lines = []
        i = 0
        while i < len(lines):
            l = lines[i]
            if len(l.strip()) > 0 and l.strip()[0] == '#':
                cnt = 1
                j = i + 1
                while j < len(lines):
                    if len(lines[j].strip()) > 0 and lines[j].strip()[0] == '#':
                        cnt += 1
                        j += 1
                    else:
                        break
                if cnt >= threshold:
                    i += cnt
                    continue
            i += 1
            new_lines.append(l)
        f.close()
        f = open(fname, 'w')
        f.writelines(new_lines)
        f.close()

    def add_blank_lines_before_class_definition(self, fname):
        f = open(fname, 'r')
        lines = f.readlines()
        new_lines = []
        for i, l in enumerate(lines):
            if 'class ' in l:
                cnt = 0
                for ii in range(i - 4, i):
                    if self.is_blank_line(lines[ii]):
                        cnt += 1
                if cnt < 2:
                    for ii in range(2 - cnt):
                        new_lines.append('\n')
                elif cnt > 2:
                    new_lines = new_lines[:2 - cnt]
            new_lines.append(l)
        if not self.is_blank_line(new_lines[-1]):
            new_lines.append('\n')
        f.close()
        f = open(fname, 'w')
        f.writelines(new_lines)
        f.close()

    def is_blank_line(self, line):
        if line.strip(' ') == '\n' or len(line.strip(' ')) == 0:
            return True
        else:
            return False

    def run(self):
        self.build_dir()
        self.copy_code()
        self.copy_tester()
        self.change_import()
        self.remove_unused_components()
        self.change_logging()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/data/programs/probe_position_correction_w_ptychonn')
    parser.add_argument('--dest', default='/data/programs/PtychoNN')
    args = parser.parse_args()

    migrator = CodeMigrator(source_dir=args.source, dest_dir=args.dest)
    migrator.run()
