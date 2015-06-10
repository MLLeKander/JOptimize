package nl.rug.joptimize.opt;

public class PrintOptObserver<ParamType extends OptParam<ParamType>> implements
        OptObserver<ParamType> {

    @Override
    public void notifyEpoch(ParamType params, double error) {
        System.out.println(error+" " +params);
    }

    @Override
    public void notifyExample(ParamType params) {
        System.out.println(params);
    }

}
