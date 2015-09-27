package nl.rug.joptimize;

import java.util.HashMap;

public class Arguments {
    
    private final HashMap<String, String> argMap = new HashMap<String, String>();
    private final String defaultArg;

    public Arguments(String[] args) {
        StringBuilder sb = new StringBuilder();
        String sep = "";

        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--")) {
                assert (args.length > i + 1);
                argMap.put(args[i].toLowerCase(), args[i + 1]);
                i++;
            } else {
                sb.append(sep);
                sb.append(args[i]);
                sep = " ";
            }
        }
        defaultArg = sb.length() > 0 ? sb.toString() : null;
    }
    
    public String keyLiteral(String key) {
        return "--"+key.toLowerCase();
    }
    
    private String get_(String key) {
        return argMap.get(keyLiteral(key));
    }

    public boolean hasArg(String key) {
        return argMap.containsKey(keyLiteral(key));
    }
    
    public String get(String key) {
        String s = get_(key);
        if (s == null) throw new IllegalArgumentException("Could not find required key: "+key);
        return s;
    }
    
    public String get(String key, String def) {
        String s = get_(key);
        return s == null ? def : s;
    }
    
    public int getInt(String key) {
        String s = get_(key);
        if (s == null) throw new IllegalArgumentException("Could not find required integer key: "+key);
        return Integer.parseInt(s);
    }
    
    public int getInt(String key, int def) {
        String s = get_(key);
        return s == null ? def : Integer.parseInt(s);
    }
    
    public long getLong(String key) {
        String s = get_(key);
        if (s == null) throw new IllegalArgumentException("Could not find required integer key: "+key);
        return Long.parseLong(s);
    }
    
    public long getLong(String key, int def) {
        String s = get_(key);
        return s == null ? def : Long.parseLong(s);
    }
    
    public double getDbl(String key) {
        String s = get_(key);
        if (s == null) throw new IllegalArgumentException("Could not find required double key: "+key);
        return Double.parseDouble(s);
    }
    
    public boolean getBool(String key) {
        String s = get_(key);
        if (s == null) throw new IllegalArgumentException("Could not find required double key: "+key);
        return Boolean.parseBoolean(s);
    }

    public boolean getBool(String key, boolean def) {
        String s = get_(key);
        return s == null ? def : Boolean.parseBoolean(s);
    }
    
    public double getDbl(String key, double def) {
        String s = get_(key);
        return s == null ? def : Double.parseDouble(s);
    }
    
    public boolean hasDefault() {
        return defaultArg != null;
    }
    
    public String getDefault() {
        if (defaultArg == null) throw new IllegalArgumentException("Could not find required default key.");
        return defaultArg;
    }
    
    public String toString() {
    	String argStr = argMap.toString().replaceAll("[{},]","").replaceAll("="," ");
    	if (hasDefault()) {
    		return argStr;
    	} else {
    		return getDefault()+" "+argStr;
    	}
    }
}
