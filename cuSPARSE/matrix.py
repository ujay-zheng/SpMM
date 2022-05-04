import numpy as np
import os
import sys


def total_random_mask(row, col, sparsity):
    mask = np.zeros(row*col)
    mask[int(row*col*sparsity)::] = 1
    np.random.shuffle(mask)
    mask = mask.reshape(row, col)
    return mask


def block_random_mask(row, col, sparsity, block_row, block_col):
    assert row % block_row == 0
    assert col % block_col == 0

    num_blocks_row = row // block_row
    num_blocks_col = col // block_col

    block_mask = total_random_mask(num_blocks_row, num_blocks_col, sparsity)
    ret = np.zeros((row, col))

    for ridx in range(0, row, block_row):
        for cidx in range(0, col, block_col):
            rblock_idx = ridx // block_row
            cblock_idx = cidx // block_col
            block_mask_val = block_mask[rblock_idx, cblock_idx]
            ret[ridx:ridx+block_row, cidx:cidx+block_col] = block_mask_val
    return ret


def generate_guide(matrix_dir, guide_path):
    for root, dirs, files in os.walk(matrix_dir, topdown=False):
        abs_path = os.path.abspath(root)
        with open(guide_path, "w") as f:
            for file in files:
                f.write(abs_path+"/"+file+"\n")
        break


def mask_pipeline(path, out_path, guide_file, br_range, bc_range, sparsity_range):
    weight_size = np.loadtxt(path, dtype=np.int64)
    for M, K, N in weight_size:
        sparse = np.random.normal(size=(M, K))
        dense = np.random.normal(size=(K, N))
        matrix_dir = '''%d_%d_%d/''' % (M, K, N)
        new_dir = out_path + matrix_dir
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        for br in br_range:
            for bc in bc_range:
                if M % br != 0:
                    break
                if K % bc != 0:
                    continue
                for sparsity in sparsity_range:
                    matrix_file = '''%d_%d_%.1f.txt''' % (br, bc, sparsity)
                    with open(new_dir + matrix_file, "w") as f:
                        f.write('''%d %d %d\n%d %d %.1f\n''' % (M, K, N, br, bc, sparsity))
                        sparse_cp = sparse.copy()
                        mask_id = block_random_mask(M, K, sparsity, br, bc)
                        sparse_cp[mask_id == 0] = 0
                        for i in range(sparse_cp.shape[0]):
                            for j in range(sparse_cp.shape[1]):
                                f.write('''%f ''' %(sparse_cp[i, j]))
                            f.write("\n")
                        f.write("\n")
                        for i in range(dense.shape[0]):
                            for j in range(dense.shape[1]):
                                f.write('''%f ''' % (dense[i, j]))
                            f.write("\n")
                        f.write("\n")
    generate_guide(out_path, guide_file)


if __name__ == '__main__':
    # np.savetxt("opSize.txt", np.array([[768, 768, 16], [3072, 758, 16], [768, 3072, 16], [32, 288, 1024], [128, 1152, 64]],
    #                                 dtype=np.int64), fmt="%i %i %i")
    # mask_pipeline("opSize.txt", "matrix/", "guide.txt", range(1, 65), range(1, 65),
    #               [0.6, 0.7, 0.8, 0.9])
    generate_guide(sys.argv[1], sys.argv[2])
