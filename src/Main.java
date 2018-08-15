import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {
    public List<Character> readCharactersFromFile(String resource) throws IOException {

        List<Character> characters = new ArrayList<>();

        //  Read iterator from characters text file
        InputStream resourceAsStream = getClass().getResourceAsStream(resource);
        BufferedReader br = new BufferedReader(new InputStreamReader(resourceAsStream));

        String line = "";

        Character c = null;

        while ((line = br.readLine()) != null) {
            if (line.length() > 0) {
                if (line.length() == 1) {
                    if (c != null) {
                        characters.add(c);
                    }

                    // Create new char since we have a first letter
                    c = new Character();

                    // Set character desired value
                    c.desired = line;
                } else {
                    c.vectors.addAll(Arrays.stream(line.split(",")).map(Double::parseDouble).collect(Collectors.toList()));
                }
            }
        }

        // Add last letter to character map
        characters.add(c);


        // Return characters
        return characters;
    }

    public static void main(String[] args) {

    }
}
