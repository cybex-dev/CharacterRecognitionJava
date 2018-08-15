import java.util.ArrayList;
import java.util.List;

public class Character {
    public String desired = "";
    public List<Double> vectors = new ArrayList<>();

    public Character() {
    }

    public Character(String desired, List<Double> vectors) {
        this.desired = desired;
        this.vectors = vectors;
    }

    public double[] getVectors() {
        return vectors.stream().mapToDouble(Double::doubleValue).toArray();
    }

    public int[] getBinary() {
        switch (desired) {
            case "A": return new int[]{0, 0, 0, 0, 1};
            case "B": return new int[]{0,0,0,1,0};
            case "C": return new int[]{0,0,0,1,1};
            case "D": return new int[]{0,0,1,0,0};
            case "E": return new int[]{0,0,1,0,1};
            case "F": return new int[]{0,0,1,1,0};
            case "G": return new int[]{0,0,1,1,1};
            case "H": return new int[]{0,1,0,0,0};
            case "I": return new int[]{0,1,0,0,1};
            case "J": return new int[]{0,1,0,1,0};
            case "K": return new int[]{0,1,0,1,1};
            case "L": return new int[]{0,1,1,0,0};
            case "M": return new int[]{0,1,1,0,1};
            case "N": return new int[]{0,1,1,1,0};
            case "O": return new int[]{0,1,1,1,1};
            case "P": return new int[]{1,0,0,0,0};
            case "Q": return new int[]{1,0,0,0,1};
            case "R": return new int[]{1,0,0,1,0};
            case "S": return new int[]{1,0,0,1,1};
            case "T": return new int[]{1,0,1,0,0};
            case "U": return new int[]{1,0,1,0,1};
            case "V": return new int[]{1,0,1,1,0};
            case "W": return new int[]{1,0,1,1,1};
            case "X": return new int[]{1,1,0,0,0};
            case "Y": return new int[]{1,1,0,0,1};
            case "Z": return new int[]{1,1,0,1,0};
            default: return new int[]{};
        }
    }
}
