import torch

class Distribution_for_entropy2(torch.nn.Module):
    def __init__(self,distribution_type = 'logistic',mixture=False):
        super(Distribution_for_entropy2, self).__init__()
        self.distribution_type =distribution_type
        self.mixture=mixture

    def forward(self, x, p_dec):
        channel = p_dec.size()[1]
        # p_dec=torch.transpose(p_dec,0,2,3,1)
        # p_dec=p_dec.transpose(1,3).transpose(1,2)
        if self.mixture:
            if channel % args.n_mixtures != 0:
                raise ValueError(
                    "channel number must be multiple of 3")
            gauss_num = channel//3
            # mean,scale,probs=torch.chunk(p_dec, gauss_num, dim=1)/
            temp = list(torch.chunk(p_dec, channel, dim=1))
            # temp=list(temp_)
            # keep the weight  summation of prob == 1
            probs = torch.cat(temp[gauss_num*2: ], dim=1)
            probs = F.softmax(probs, dim=-1)

            # 0-k-1:mean  k:k*2-1: scale k*2:-1： weights
            # process the scale value to non-zero  如果为0，就设为最小值1*e-6
            # for i in range(gauss_num, gauss_num*2):
            #     temp[i] = torch.abs(temp[i])
            #     temp[i][temp[i] == 0] = 1e-6

            gauss_list = []
            likelihoods = 0
            likelihood_list = []
            if self.distribution_type=='laplace':
                for i in range(gauss_num, gauss_num*2):
                    temp[i] = torch.abs(temp[i])
                    temp[i][temp[i] == 0] = 1e-6
                for i in range(gauss_num):
                # gauss_list.append(torch.distributions.normal.Normal(temp[i], temp[i+gauss_num]))
                    gauss_list.append(torch.distributions.laplace.Laplace(temp[i], temp[i+gauss_num]))
                    likelihood_list.append(torch.abs(gauss_list[i].cdf(x + 0.5/255.)-gauss_list[i].cdf(x-0.5/255.)))
                    # likelihood_list.append(F.sigmoid(x + 0.5/255.)-F.sigmoid(x-0.5/255.))
            elif self.distribution_type=='normal':
                for i in range(gauss_num, gauss_num*2):
                    temp[i] = torch.abs(temp[i])
                    temp[i][temp[i] == 0] = 1e-6
                for i in range(gauss_num):
                    gauss_list.append(torch.distributions.normal.Normal(temp[i], temp[i+gauss_num]))
                    likelihood_list.append(torch.abs(gauss_list[i].cdf(x + 0.5/255.)-gauss_list[i].cdf(x-0.5/255.)))

            else:
                # likelihoods=compute_log_pz((mean,scale,probs),x,args)
                for i in range(gauss_num):
                    # gauss_list.append(torch.distributions.(temp[i], temp[i+gauss_num]))
                    likelihood_list.append(F.sigmoid((x + 0.5/255.-temp[i])/torch.exp(temp[i+gauss_num]))-F.sigmoid((x - 0.5/255.-temp[i])/torch.exp(temp[i+gauss_num])))
                # likelihoods=compute_log_pz((mean,scale,probs),x,args)
            # likelihoods = 0
            for i in range(gauss_num):
                likelihoods += probs[:,i:i+1,:,:] * likelihood_list[i]
        else:
            mean,scales=torch.chunk(p_dec, channel, dim=1)
            # 0-k-1:mean  k:k*2-1: scale k*2:-1： weights
            # process the scale value to non-zero  如果为0，就设为最小值1*e-6
            scales = torch.abs(scales)
            scales[scales == 0] = 1e-6
            if self.distribution_type=='laplace':
                scales = torch.abs(scales)
                scales[scales == 0] = 1e-6
                gauss_list=torch.distributions.laplace.Laplace(mean, scales)
                likelihoods=torch.abs(gauss_list.cdf(x + 0.5/255.)-gauss_list.cdf(x-0.5/255.))
                    # likelihood_list.append(F.sigmoid(x + 0.5/255.)-F.sigmoid(x-0.5/255.))
            elif self.distribution_type=='normal':
                scales = torch.abs(scales)
                scales[scales == 0] = 1e-6
                gauss_list=torch.distributions.normal.Normal(mean, scales)
                likelihoods=torch.abs(gauss_list.cdf(x + 0.5/255.)-gauss_list.cdf(x-0.5/255.))

            else:
                # likelihoods=compute_log_pz((mean,scale,probs),x,args)
                likelihoods=F.sigmoid((x + 0.5/255.-mean)/torch.exp(scales))-F.sigmoid((x - 0.5/255.-mean)/torch.exp(scales))
                # likelihoods=compute_log_pz((mean,scale,probs),x,args)
        return likelihoods