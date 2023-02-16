Test Markdown document
======================

Text
----

Here is a paragraph with bold text. **This is some bold text.** Here is a
paragraph with bold text. __This is also some bold text.__

Here is another one with italic text. *This is some italic text.* Here is
another one with italic text. _This is some italic text._

Here is another one with struckout text. ~~This is some struckout text.~~

LSTM V2 : 
Cross entropy loss for the test set 

[1.2768334 1.2795261 1.2780919 1.2786841 1.2774734 1.2772642 1.2829635
 1.2858839 1.2774107 1.2817495 1.3638328 1.2767655 1.2774628 1.2780952
 1.2763233 1.2766747 1.2807722 1.5570608 1.2768    1.5058396] 
accuracy_score 1.0


Links
-----

Autolink: <http://example.com>

Link: [Example](http://example.com)

Reference style [link][1].

[1]: http://example.com  "Example"


Images
------

Image: ![My image](http://www.foo.bar/image.png)

Headers
-------

# First level title
## Second level title
### Third level title
#### Fourth level title
##### Fifth level title
###### Sixth level title

### Title with [link](http://localhost)
### Title with ![image](http://localhost)

Code
----

```
This
  is
    code
      fence
```

Inline `code span in a` paragraph.

This is a code block:

    /**
     * Sorts the specified array into ascending numerical order.
     *
     * <p>Implementation note: The sorting algorithm is a Dual-Pivot Quicksort
     * by Vladimir Yaroslavskiy, Jon Bentley, and Joshua Bloch. This algorithm
     * offers O(n log(n)) performance on many data sets that cause other
     * quicksorts to degrade to quadratic performance, and is typically
     * faster than traditional (one-pivot) Quicksort implementations.
     *
     * @param a the array to be sorted
     */
    public static void sort(byte[] a) {
        DualPivotQuicksort.sort(a);
    }

Quotes
------

> This is the first level of quoting.
>
> > This is nested blockquote.
>
> Back to the first level.


> A list within a blockquote:
>
> *	asterisk 1
> *	asterisk 2
> *	asterisk 3


> Formatting within a blockquote:
>
> ### header
> Link: [Example](http://example.com)



Html
-------

This is inline <span>html</html>.
And this is an html block.

<table>
  <tr>
    <th>Column 1</th>
    <th>Column 2</th>
  </tr>
  <tr>
    <td>Row 1 Cell 1</td>
    <td>Row 1 Cell 2</td>
  </tr>
  <tr>
    <td>Row 2 Cell 1</td>
    <td>Row 2 Cell 2</td>
  </tr>
</table>

Horizontal rules
----------------

---

___


***


Lists
-----

Unordered list:

*	asterisk 1
*	asterisk 2
*	asterisk 3


Ordered list:

1.	First
2.	Second
3.	Third


Mixed:

1. First
2. Second:
	* Fee
	* Fie
	* Foe
3. Third


Tables:

| Header 1 | Header 2 |
| -------- | -------- |
| Data 1   | Data 2   |