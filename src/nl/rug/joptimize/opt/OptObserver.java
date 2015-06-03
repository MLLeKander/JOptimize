package nl.rug.joptimize.opt;

public interface OptObserver {
    // TODO
    public void notifyEpoch(OptParam params);

    public void notifyExample(OptParam params);
}
