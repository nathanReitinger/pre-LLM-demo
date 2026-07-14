// corpus.js
// An original, hand-written background corpus used purely to give the
// n-gram models enough word-frequency data to back off to when the
// story text alone is too sparse. None of this text is quoted from any
// existing work — it's generic filler prose written for this project.

const BACKGROUND_CORPUS = `
The old house stood at the end of the road, quiet and still in the morning light.
Every year the town held a small fair near the river, and everyone who lived there went.
She walked to school with her brother, talking about the weekend and what they would do next.
The teacher asked a question, and the whole class waited to see who would answer first.
It was raining again, and the students ran across the yard to get out of the storm.
Their mother called them in for dinner, and they came running, hungry after a long day outside.
The team practiced every afternoon, hoping that this season would finally be the one they won.
He opened the door slowly, not sure what he would find on the other side.
The garden behind the house was full of flowers, and the smell of them drifted through the window.
Nobody expected the meeting to run so long, but the committee had a great deal to discuss.
The coach gathered the players together and told them how proud he was of their effort.
She practiced the piano every evening, even when she was tired and wanted to stop.
The letter arrived on a Tuesday, and no one in the family knew what to expect from it.
The two friends had known each other since they were children, and they trusted one another completely.
It was the kind of afternoon that made everyone want to be outside instead of indoors.
The principal walked into the room and asked for silence before the announcement began.
The dog ran down the street chasing after the ball that had rolled into the yard.
When the bell rang, the students gathered their books and hurried toward the door.
The town had never seen a storm quite like the one that hit that spring.
Her sister had always been the confident one, the first to raise a hand in class.
The crowd grew quiet as the speaker stepped up to the microphone and cleared her throat.
Everyone in the gym turned to look when the name was finally called.
The committee had debated for weeks about who should be chosen to speak.
He was nervous, but he walked to the front of the room anyway.
She had rehearsed the speech a hundred times in front of the mirror at home.
The audience clapped as the student walked slowly toward the stage.
Nobody in the family had ever given a speech in front of so many people.
The two sisters looked nothing alike, but they were closer than anyone else in town.
It surprised everyone when the quiet one stepped forward instead of her sister.
The room fell silent, and then the applause began, louder than anyone expected.
The teacher smiled and nodded, proud of the student standing at the podium.
Parents in the audience leaned forward, eager to hear what would be said next.
The younger student felt her hands shake as she unfolded the piece of paper.
It had been a long year, full of changes that nobody could have predicted.
The whole town seemed to hold its breath as the ceremony began.
Friends in the crowd whispered to each other, wondering who would be chosen.
The gymnasium was decorated with banners and balloons for the graduation.
Somewhere in the back row, a parent wiped away a tear before the speech began.
The principal read the name twice, just to make sure everyone had heard it clearly.
It was not the name most people in the room had been expecting.
`;

if (typeof module !== 'undefined') {
  module.exports = { BACKGROUND_CORPUS };
}
