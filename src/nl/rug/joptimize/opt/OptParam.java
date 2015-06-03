package nl.rug.joptimize.opt;

public interface OptParam {
	public OptParam add(OptParam o);

	public OptParam add_s(OptParam o);

	public OptParam sub(OptParam o);

	public OptParam sub_s(OptParam o);

	public OptParam zero();

	public OptParam zero_s();

	public OptParam dotprod(OptParam o);

	public OptParam dotprod_s(OptParam o);

	public OptParam multiply(double o);

	public OptParam multiply_s(double o);
}
