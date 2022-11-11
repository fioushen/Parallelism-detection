
import logging
logger = logging.getLogger('main.tools')


class Parameters:


    def __init__(self, **args):
        self.__dict__.update(args)  # 将参数加入到self中

    def __add__(self, hps):

        if not isinstance(hps, Parameters):
            raise Exception(f'{type(self)} and {type(hps)} cannot be added together！！！ --- by LIC ')
        param_dict = dict()
        param_dict.update(self.__dict__)
        param_dict.update(hps.__dict__)
        return Parameters(** param_dict)

    def to_str(self):

        params = sorted(self.__dict__.items(), key=lambda item: item[0])
        output = ''
        for param, value in params:
            output += f'{param:18}  {value}\n'
        return output

    def __str__(self):
        return self.to_str()

    def to_dict(self):

        return self.__dict__

    def get(self, attr_name):

        return self.__dict__.get(attr_name)

    def set(self, key, value):

        self.__dict__[key] = value

    def default(self, default_params):

        for key, value in default_params.items():
            if self.__dict__.get(key) is None:
                self.__dict__[key] = value

    def update(self, params):

        param_set = params
        if isinstance(params, Parameters):
            param_set = params.to_dict()
        for key, value in param_set.items():
            if self.__dict__.get(key) is not None:
                self.__dict__[key] = value

    def extend(self, params):

        if isinstance(params, Parameters):
            params = params.to_dict()
        self.__dict__.update(params)
