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
}
