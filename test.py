from ollama import Client

from main import generate_data, mock_llm, OllamaLLMWrapper

# Sample schema
schema_example = {
    "name": "quiz",
    "attributes": [
        {
            "name": "questions",
            "queryString": "Create a list of questions from the given text: #{_chunk}\n\nOnly include "
                           "questions that can be answered from the given text. This "
                           "will be processed automatically so your formatting is very important. Don't include "
                           "text other than the list of questions. Do not include the answer.",
            "listType": {
                "attributes": [
                    {"name": "questionText"},
                    {
                        "name": "answers",
                        "queryString": "Question basis text:\n#{_parent._parent._chunk}\n\nBased on the "
                                       "above text, provide possible answers for the quiz question: "
                                       "#{questionText}\nThis will be processed automatically so your "
                                       "formatting is very important. Do not include text other than the list of "
                                       "answers. Do not repeat the question.",
                        "listType": {
                            "type": "string"
                        }
                    }
                ]
            }
        }
    ]
}


def test_mock_llm():
    return generate_data(schema_example, mock_llm, "{sample text}")


def test_ollama():
    return generate_data(schema_example, OllamaLLMWrapper(Client(host="http://127.0.0.1:11434"), 'llama3.2'), """
The Great Molasses Flood, also known as the Boston 
Molasses Disaster,[1][2][a] was a disaster that occurred on Wednesday, January 15, 1919, in the North End 
neighborhood of Boston, Massachusetts.

A large storage tank filled with 2.3 million U.S. gallons (8,700 cubic meters)[4] of molasses, 
weighing approximately[b] 13,000 short tons (12,000 metric tons) burst, and the resultant wave of molasses rushed 
through the streets at an estimated 35 miles per hour (56 kilometers per hour), killing 21 people and injuring 150.[
5] The event entered local folklore and residents reported for decades afterwards that the area still smelled of 
molasses on hot summer days.[5][6]

Flood The front page of an old newspaper. The headline reads, "HUGE MOLASSES TANK EXPLODES IN NORTH END; 11 DEAD, 
50 HURT". Coverage from The Boston Post Molasses can be fermented to produce ethanol, the active ingredient in 
alcoholic beverages and a key component in munitions.[7]: 11  The disaster occurred at the Purity Distilling Company 
facility at 529 Commercial Street near Keany Square. A considerable amount of molasses had been stored there by the 
company, which used the harborside Commercial Street tank to offload molasses from ships and store it for later 
transfer by pipeline to the Purity ethanol plant situated between Willow Street and Evereteze Way in Cambridge, 
Massachusetts. The molasses tank stood 50 feet (15 meters) tall and 90 ft (27 m) in diameter, and contained as much 
as 2.3 million US gal (8,700 m3).

A scanned color map. The area around North End Beach and Charlestown Bridge is circled in red. Modern downtown Boston 
with molasses flood area circled On January 15, 1919, temperatures in Boston had risen above 40 degrees Fahrenheit (4 
degrees Celsius), climbing rapidly from the frigid temperatures of the preceding days,[7]: 91, 95  and the previous 
day, a ship had delivered a fresh load of molasses, which had been warmed to decrease its viscosity for transfer.[8] 
Possibly due to the thermal expansion of the older, colder molasses already inside the tank, the tank burst open and 
collapsed at approximately 12:30 p.m. Witnesses reported that they felt the ground shake and heard a roar as it 
collapsed, a long rumble similar to the passing of an elevated train; others reported a tremendous crashing, 
a deep growling, "a thunderclap-like bang!", and a sound like a machine gun as the rivets shot out of the tank.[
7]: 92–95 

The density of molasses is about 1.4 metric tons per cubic meter (12 pounds per US gallon), 40 percent more dense 
than water, resulting in the molasses having a great deal of potential energy.[9] The collapse translated this energy 
into a wave of molasses 25 ft (8 m) high at its peak,[10] moving at 35 mph (56 km/h).[5][6] The wave was of 
sufficient force to drive steel panels of the burst tank against the girders of the adjacent Boston Elevated 
Railway's Atlantic Avenue structure[11] and tip a streetcar momentarily off the El's tracks. Stephen Puleo describes 
how nearby buildings were swept off their foundations and crushed. Several blocks were flooded to a depth of 2 to 3 
ft (60 to 90 cm). Puleo quotes a Boston Post report:

Molasses, waist deep, covered the street and swirled and bubbled about the wreckage [...] Here and there struggled a 
form—whether it was animal or human being was impossible to tell. Only an upheaval, a thrashing about in the sticky 
mass, showed where any life was [...] Horses died like so many flies on sticky fly-paper. The more they struggled, 
the deeper in the mess they were ensnared. Human beings—men and women—suffered likewise.[7]: 98 

The Boston Globe reported that people "were picked up by a rush of air and hurled many feet". Others had debris 
hurled at them from the rush of sweet-smelling air. A truck was picked up and hurled into Boston Harbor. After the 
initial wave, the molasses became viscous, exacerbated by the cold temperatures, trapping those caught in the wave 
and making it even more difficult to rescue them.[9] About 150 people were injured, and 21 people and several horses 
were killed. Some were crushed and drowned by the molasses or by the debris that it carried within.[12] The wounded 
included people, horses, and dogs; coughing fits became one of the most common ailments after the initial blast. 
Edwards Park wrote of one child's experience in a 1983 article for Smithsonian:

Anthony di Stasio, walking homeward with his sisters from the Michelangelo School, was picked up by the wave and 
carried, tumbling on its crest, almost as though he were surfing. Then he grounded and the molasses rolled him like a 
pebble as the wave diminished. He heard his mother call his name and couldn't answer, his throat was so clogged with 
the smothering goo. He passed out, then opened his eyes to find three of his four sisters staring at him.[6]

Aftermath

Damage to the Boston Elevated Railway caused by the burst tank and resulting flood First to the scene were 116 cadets 
under the direction of Lieutenant Commander H. J. Copeland from USS Nantucket, a training ship of the Massachusetts 
Nautical School (now the Massachusetts Maritime Academy) that was docked nearby at the playground pier.[13] The 
cadets ran several blocks toward the accident and entered into the knee-deep flood of molasses to pull out the 
survivors, while others worked to keep curious onlookers from getting in the way of the rescuers. The Boston Police, 
Red Cross, Army, and Navy personnel soon arrived. Some nurses from the Red Cross dove into the molasses, while others 
tended to the injured, keeping them warm and feeding the exhausted workers. Many of these people worked through the 
night, and the injured were so numerous that doctors and surgeons set up a makeshift hospital in a nearby building. 
Rescuers found it difficult to make their way through the syrup to help the victims, and four days elapsed before 
they stopped searching; many of the dead were so glazed over in molasses that they were hard to recognize.[6] Other 
victims were swept into Boston Harbor and were found three to four months after the disaster.[12]

In the wake of the accident, 119 residents brought a class-action lawsuit against the United States Industrial 
Alcohol Company (USIA),[14] which had bought Purity Distilling in 1917. It was one of the first class-action suits in 
Massachusetts and is considered a milestone in paving the way for modern corporate regulation.[15] The company 
claimed that the tank had been blown up by anarchists[7]: 165  because some of the alcohol produced was to be used in 
making munitions, but a court-appointed auditor found USIA responsible after three years of hearings, and the company 
ultimately paid out $628,000 in damages[15] ($11 million in 2023, adjusted for inflation[16]). Relatives of those 
killed reportedly received around $7,000 per victim (equivalent to $123,000 in 2023).[6]

Cleanup 

Cleanup crews used salt water from a fireboat to wash away the molasses and sand to absorb it,[17] and the 
harbor was brown with molasses until summer.[18] The cleanup in the immediate area took weeks,[19] with several 
hundred people contributing to the effort,[7]: 132–134, 139 [15] and it took longer to clean the rest of Greater 
Boston and its suburbs. Rescue workers, cleanup crews, and sight-seers had tracked molasses through the streets and 
spread it to subway platforms, to the seats inside trains and streetcars, to pay telephone handsets, into homes,
[6][7]: 139  and to countless other places. It was reported that "Everything that a Bostonian touched was sticky."[
6]""")


if __name__ == "__main__":
    print(f'Mock LLM Test: \n{test_mock_llm()}')
    print('\n')
    print(f'Ollama Test: \n{test_ollama()}')
