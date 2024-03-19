import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1d_1x3(in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1, bias=True):
    """ 1x3 convolution """
    return nn.Conv1d(in_channels=in_ch, out_channels=out_ch,\
                     kernel_size=kernel_size, stride=stride, padding=padding,\
                     groups=groups, bias=bias)

def conv1d_1x1(in_ch, out_ch, stride=1, bias=True):
    """ 1x1 convolution """
    return nn.Conv1d(in_channels=in_ch, out_channels=out_ch,\
                     kernel_size=1, stride=stride, padding=0, bias=bias)

def conv2d_3x3(in_ch, out_ch, stride=1, groups=1, dilation=1, bias=True):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,\
                     padding=dilation, groups=groups, bias=bias,\
                     dilation=dilation)

def conv2d_1x1(in_ch, out_ch, stride=1, groups=1, dilation=1, bias=True):
    """ 1x1 convolution with padding """
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=bias)

def fc(in_dim, out_dim, bias=True):
    return nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)

def bn1d(num_features, eps=1e-05, momentum=0.1, affine=True):
    """BatchNorm1d"""
    return nn.BatchNorm1d(num_features=num_features,\
                          eps=eps,\
                          momentum=momentum,\
                          affine=affine)

def bn2d(num_features, eps=1e-05, momentum=0.1, affine=True):
    """BatchNorm2d"""
    return nn.BatchNorm2d(num_features=num_features,\
                          eps=eps, momentum=momentum, affine=affine)

def gn(num_groups, num_channels, eps=1e-05, affine=True):
    """GroupNorm"""
    return nn.GroupNorm(num_groups, num_channels=num_channels,\
                        eps=eps, affine=affine)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class convlstmcell(nn.Module):
    """
    :ref https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    :ref https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
    """
    def __init__(self, channels, c_hidden, kernel_size=(3, 3), bias=True,
            with_normalization=False,
            # normlization_modality='LayerNorm',
            normlization_modality='GroupNorm',
        ):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        channels: int
            Number of channels of input tensor.
        c_hidden: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(convlstmcell, self).__init__()

        self.c_hidden = c_hidden
        self.with_normalization = with_normalization

        self.gate = nn.Sequential(
            nn.Conv2d(channels + c_hidden, 4 * c_hidden,
                bias=bias,
                kernel_size=kernel_size,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            ),
        )

        if self.with_normalization:
            if normlization_modality == 'LayerNorm':
                self.norm_gate = nn.LayerNorm([4 * c_hidden, hw[0], hw[1]])
                self.norm_prev_cell = nn.LayerNorm([c_hidden, hw[0], hw[1]])
                self.norm_curr_cell = nn.LayerNorm([c_hidden, hw[0], hw[1]])
            elif normlization_modality == 'BatchNorm':
                self.norm_gate = nn.BatchNorm2d(4 * c_hidden)
                self.norm_cell = nn.BatchNorm2d(c_hidden)
            elif normlization_modality == 'GroupNorm':
                self.norm_gate = nn.GroupNorm(64, 4 * c_hidden)
                self.norm_prev_cell = nn.GroupNorm(16, c_hidden)
                self.norm_curr_cell = nn.GroupNorm(16, c_hidden)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, prev_state):
        prev_hidden, prev_cell = prev_state

        stacked_inputs = torch.cat((x, prev_hidden), dim=1)
        gates = self.gate(stacked_inputs) # [bz, 4*c_hidden, H, W]
        if self.with_normalization:
            gates = self.norm_gate(gates)

        """ chunk across channel dimension """
        in_gate, rmr_gate, out_gate, cell_gate = gates.chunk(4, 1)

        """ apply sigmoid non linearity """
        in_gate = self.sigmoid(in_gate) # [bz, C=c_hidden, H, W]
        rmr_gate = self.sigmoid(rmr_gate) # [bz, C=c_hidden, H, W]
        out_gate = self.sigmoid(out_gate) # [bz, C=c_hidden, H, W]

        """ apply hypertangent non linearity """
        cell_gate = self.tanh(cell_gate) # [bz, C=Hidden, H, W]

        """ compute current cell and hidden state """
        if self.with_normalization:
            # cell = self.norm_cell(rmr_gate * prev_cell) + (in_gate * cell_gate) # [bz, C', H, W]
            cell = self.norm_prev_cell(rmr_gate * prev_cell) + self.norm_curr_cell(in_gate * cell_gate) # [bz, C', H, W]
        else:
            cell = (rmr_gate * prev_cell) + (in_gate * cell_gate) # [bz, C', H, W]
        hidden = out_gate * self.tanh(cell)

        return hidden, cell

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.c_hidden, height, width,
                device=self.gate[0].weight.device),
            torch.zeros(batch_size, self.c_hidden, height, width,
                device=self.gate[0].weight.device)
        )

class convlstm(torch.nn.Module):
    """

    Parameters:
        channels: int
            Number of channels in input
        c_hidden: list
            Number of hidden channels
        kernel_size: (int, int)
            Size of kernel in convolutions
        num_layers: int
            Number of LSTM layers stacked on each other
        batch_first: bool
            Whether or not dimension 0 is the batch or not
        bias: bool
            Bias or no bias in Convolution
        return_all_layers: bool
            Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size [B, T, C, H, W] or [T, B, C, H, W]
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = convlstm(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, channels, c_hidden, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, **kwargs):
        super(convlstm, self).__init__()
        self.with_residual = kwargs.get('with_residual', False)
        self.with_normalization = kwargs.get('with_normalization', False)

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        c_hidden = self._extend_for_multilayer(c_hidden, num_layers)

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for l in range(0, self.num_layers):
            cur_input_dim = channels if l == 0 else c_hidden[l - 1]

            cell_list.append(
                convlstmcell(channels=cur_input_dim,
                             c_hidden=c_hidden[l],
                             kernel_size=kernel_size[l],
                             with_normalization=self.with_normalization,
                             bias=bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state=None):
        """

        Parameters
        ----------
        x:
            5-D Tensor either of shape [ns, bz, x_c, x_h, x_w] or [bz, ns, x_c, x_h, x_w]
        hidden_state:
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """

        if not self.batch_first:
            # [ns, bz, x_c, x_h, x_w] -> [bz, ns, x_c, x_h, x_w]
            x = x.permute(1, 0, 2, 3, 4)

        bz, _,  _, x_h, x_w = x.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=bz,
                                             image_size=(x_h, x_w))

        layer_output_list = []
        last_state_list = []

        seq_len = x.size(1)
        cur_layer_input = x

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :],
                                                 prev_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1) # [bz, ns, C, H=8, W=8]
            cur_layer_input = layer_output

            if self.batch_first:
                layer_output_list.append(layer_output)
            else:
                layer_output_list.append(layer_output.permute(1, 0, 2, 3, 4)) # [ns, bz, C, H, W]
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
