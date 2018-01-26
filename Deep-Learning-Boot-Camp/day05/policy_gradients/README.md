# bootcamp

### What this is

A quick PyTorch example for training a policy gradient network over the CartPole game from OpenAI's gym. This is mostly stolen from PyTorch's own tutorial, with modifications to taste for the Israel DL Bootcamp.

### Running

This requires first running `prerequisites.sh`, followed by the `run.sh` command. The `run.sh` is nothing special, it simply runs jupyer as 
```bash
xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --allow-root
```

Enjoy.
