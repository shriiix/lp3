import heapq
def huffman_encoding(data):
    frequency = {char: data.count(char) for char in set(data)}
    priority_queue = [[freq, [char, ""]] for char, freq in frequency.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        for pair in left[1:]:
            pair[1] = '0' + pair[1]
        for pair in right[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(priority_queue, [left[0] + right[0]] + left[1:] + right[1:])

    huffman_codes = dict(pair for pair in priority_queue[0][1:])
    encoded_data = ''.join([huffman_codes[char] for char in data])

    return encoded_data, huffman_codes

# Example usage
data = input("Enter String: ")
encoded_data, huffman_codes = huffman_encoding(data)
print("Encoded Data:", encoded_data)
print("Huffman Codes:", huffman_codes)