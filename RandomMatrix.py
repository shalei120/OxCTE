import torch


class RandomMatrix(object):
    r"""Abstract class providing a method for constructing 2D random arrays.
    Class is responsible for constructing 2D random arrays.
    """

    def get_2d_array(self, nb_rows, nb_columns):
        raise NotImplementedError('Abstract method')


class GaussianUnstructuredRandomMatrix(RandomMatrix):

    def __init__(self):
        pass

    def get_2d_array(self, nb_rows, nb_columns):
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        return torch.randn((self.nb_rows, self.nb_columns))


class GaussianOrthogonalRandomMatrix(RandomMatrix):
    r"""Class providing a method to create Gaussian orthogonal matrix.
    Class is responsible for constructing 2D Gaussian orthogonal arrays.
    """

    def __init__(self, scaling=0):
        self.scaling = scaling

    def get_2d_array(self, nb_rows, nb_columns):
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        nb_full_blocks = int(self.nb_rows / self.nb_columns)
        block_list = []
        # rng = self.key
        for _ in range(nb_full_blocks):
            # rng, rng_input = jax.random.split(rng)
            unstructured_block = torch.randn((self.nb_columns, self.nb_columns))
            q, _ = torch.qr(unstructured_block)
            q = torch.transpose(q, 0, 1)
            block_list.append(q)
        remaining_rows = self.nb_rows - nb_full_blocks * self.nb_columns
        if remaining_rows > 0:
            # rng, rng_input = jax.random.split(rng)
            unstructured_block = torch.randn((self.nb_columns, self.nb_columns))
            q, _ = torch.qr(unstructured_block)
            q = torch.transpose(q, 0,1)
            block_list.append(q[0:remaining_rows])
        # print([m.size() for m in block_list])
        final_matrix = torch.cat(block_list, dim = 0)

        if self.scaling == 0:
            multiplier = torch.norm(final_matrix, dim=1)
        elif self.scaling == 1:
            multiplier = torch.sqrt(float(self.nb_columns)) * torch.ones((self.nb_rows))
        else:
            raise ValueError('Scaling must be one of {0, 1}. Was %s' % self._scaling)

        return torch.matmul(torch.diag(multiplier), final_matrix)
