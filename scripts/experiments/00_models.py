import sys

sys.path.append("../..")

from _base_models import persistence, regression, linear_nn, rnn, earnn

# if __name__ == "__main__":
#     always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb", "E", "p0001"]
#     important_vars = ["VCI", "precip", "t2m", "pev", "p0005", "SMsurf", "SMroot"]

#     persistence()
#     regression(ignore_vars=always_ignore_vars)
#     linear_nn(ignore_vars=always_ignore_vars)
#     rnn(ignore_vars=always_ignore_vars)
#     earnn(ignore_vars=always_ignore_vars)
#     gbdt(ignore_vars=always_ignore_vars)
