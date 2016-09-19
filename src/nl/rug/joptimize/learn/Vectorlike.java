package nl.rug.joptimize.learn;

public interface Vectorlike {
    public void set(int ndx, double value);

    public double get(int ndx);

    public int length();

}