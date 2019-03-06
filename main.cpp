#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <optional>
#include <set>
#include <fstream>

#ifndef VERT_SHADER_NAME
    #define VERT_SHADER_NAME "vert.spv"
#endif

#ifndef FRAG_SHADER_NAME
    #define FRAG_SHADER_NAME "frag.spv"
#endif

#ifndef TEX_IMG_NAME
    #define TEX_IMG_NAME "statue.jpg"
#endif


class HelloTriangleApplication
{
public:
    void run ()
    {
        initWindow ();
        initVulkan ();
        mainLoop ();
        cleanup ();
    }

private:
    // Camera struct with MVP matrices
    // Aligned to 16 bytes per Vulkan spec
    struct Camera
    {
        alignas(16) glm::mat4 M;
        alignas(16) glm::mat4 V;
        alignas(16) glm::mat4 P;
    };

    // Simple vertex data struct
    struct Vertex
    {
        glm::vec2 pos;
        glm::vec2 tex;
        glm::vec3 color;
    };

    // Struct to store Vulkan queue family
    // with similar graphics and presentation support
    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete ()
        {
            return graphicsFamily.has_value () && presentFamily.has_value ();
        }
    };

    // Search for a queue family matching QueueFamilyIndices' criterion
    QueueFamilyIndices findQueueFamilies (VkPhysicalDevice device)
    {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilyCount; i++)
        {
            if (queueFamilies[i].queueCount > 0 && queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR (device, i, surface, &presentSupport);

            if (queueFamilies[i].queueCount > 0 && presentSupport)
                indices.presentFamily = i;

            if (indices.isComplete ())
                break;

            i++;
        }

        return indices;
    }

    // copy a buffer of memory into a VkImage
    void copyBufferToImage (VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands ();

        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { width, height, 1 };

        vkCmdCopyBufferToImage (
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

        endSingleTimeCommands (commandBuffer);
    }

    // transition the layout of an image
    // different layouts for different scenarious, i.e device copies or shader reads
    void transitionImageLayout (VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;

        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            // when we go from host to device, device src access is irrelevant but dst write access is required
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            // otherwise we're going from device memory to something the frag shader can read
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else
        {
            throw std::invalid_argument ("unsupported layout transition!");
        }

        // Single command queue to a pipeline stage transition
        VkCommandBuffer commandBuffer = beginSingleTimeCommands ();

        vkCmdPipelineBarrier (
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommands (commandBuffer);
    }

    // create a Vulkan image per the specifications
    void createImage (uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage* outImage, VkDeviceMemory* outMemory)
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

        if (vkCreateImage (logicalDevice, &imageInfo, nullptr, outImage))
            throw std::runtime_error ("failed to create image!");

        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements (logicalDevice, *outImage, &memReqs);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = findMemoryType (memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory (logicalDevice, &allocInfo, nullptr, outMemory) != VK_SUCCESS)
            throw std::runtime_error ("failed to allocate image memory!");

        vkBindImageMemory (logicalDevice, *outImage, *outMemory, 0);
    }

    // create an image view for the supplied image
    VkImageView createImageView (VkImage image, VkFormat format)
    {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView (logicalDevice, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
            throw std::runtime_error ("failed to create image view!");

        return imageView;
    }

    // allocate a buffer of memory (two structs to fill - VKBuffer and VkBufferMemory)
    void createBuffer (VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer* outBuffer, VkDeviceMemory* outMemory)
    {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer (logicalDevice, &bufferInfo, nullptr, outBuffer) != VK_SUCCESS)
            throw std::runtime_error ("failed to create buffer!");

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements (logicalDevice, *outBuffer, &memReqs);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = findMemoryType (memReqs.memoryTypeBits, properties);
        if (vkAllocateMemory (logicalDevice, &allocInfo, nullptr, outMemory) != VK_SUCCESS)
            throw std::runtime_error ("failed to allocate buffer memory!");

        vkBindBufferMemory (logicalDevice, *outBuffer, *outMemory, 0);
    }

    // use our command queue to execute a buffer copy
    void copyBuffer (VkBuffer srcBuf, VkBuffer dstBuf, VkDeviceSize size)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands ();

        VkBufferCopy copyRegion = {};
        copyRegion.size = size;
        vkCmdCopyBuffer (commandBuffer, srcBuf, dstBuf, 1, &copyRegion);

        endSingleTimeCommands (commandBuffer);
    }

    // find our device's memory type index best for type and properties
    uint32_t findMemoryType (uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties (physicalDevice, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
            if (typeFilter & (1 << i))
                if ((memProps.memoryTypes[i].propertyFlags & properties) == properties)
                    return i;

        throw std::runtime_error ("failed to find suitable memory type!");
        return 0;
    }

    // utility function to begin a command queue 
    // that executes a single command
    VkCommandBuffer beginSingleTimeCommands ()
    {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool; // always use same command pool... for now
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers (logicalDevice, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer (commandBuffer, &beginInfo);

        return commandBuffer;
    }

    // end our single time command queue 
    // (does it matter that it's a one time use?)
    void endSingleTimeCommands (VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer (commandBuffer);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit (graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle (graphicsQueue);

        vkFreeCommandBuffers (logicalDevice, commandPool, 1, &commandBuffer);
    }

    // struct to cache a physical device's swap chain capabilities
    struct SwapChainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    // fill in the above struct for a physical device
    SwapChainSupportDetails querySwapChainSupport (VkPhysicalDevice device)
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR (device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR (device, surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize (formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR (device, surface, &formatCount, details.formats.data ());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR (device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize (presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR (device, surface, &presentModeCount, details.presentModes.data ());
        }

        return details;
    }

    // choose the beset available surface format
    VkSurfaceFormatKHR chooseSwapSurfaceFormat (const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        // undefined? use 8 bit SBGR
        if (availableFormats.size () == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
        {
            return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
        }
        // otherwise try and find that format
        for (const auto& availableFormat : availableFormats)
        {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return availableFormat;
            }
        }

        // not found? use the first available format
        return availableFormats.front ();
    }

    // choose a present mode for the swap chain
    VkPresentModeKHR chooseSwapPresentMode (std::vector<VkPresentModeKHR> availablePresentModes)
    {
        // prefer mailbox for performance, use FIFO otherwise (always supported)
        auto it = std::find (availablePresentModes.begin (), availablePresentModes.end (), VK_PRESENT_MODE_MAILBOX_KHR);
        return (it == availablePresentModes.end ()) ? VK_PRESENT_MODE_FIFO_KHR : *it;
    }

    // use GLFW to get our window's extent (which we'll use to size surfaces)
    VkExtent2D chooseSwapExtent (const VkSurfaceCapabilitiesKHR* capabilities)
    {
        if (capabilities->currentExtent.width == std::numeric_limits<uint32_t>::max ())
            return capabilities->currentExtent;

        int width, height;
        glfwGetFramebufferSize (window, &width, &height);

        return { (uint32_t) width, (uint32_t) height };
    }

    // read a binary file with the STL
    static std::vector<char> readFile (const std::string& fileName)
    {
        std::ifstream file (fileName, std::ios::ate | std::ios::binary);
        if (!file.is_open ())
            throw std::runtime_error ("failed to open file " + fileName);

        size_t fileSize = (size_t) file.tellg ();
        std::vector<char> fileBuf (fileSize);
        file.seekg (0);
        file.read (fileBuf.data (), fileSize);

        return fileBuf;
    }

    // read pre-compiled shader binary SPV files
    VkShaderModule createShaderModule (const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size ();
        createInfo.pCode = (uint32_t*) code.data ();

        VkShaderModule shaderModule;
        if (vkCreateShaderModule (logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
            throw std::runtime_error ("failed to create shader module!");

        return shaderModule;
    }

    // validation layers
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_LUNARG_standard_validation"
    };

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

    // ensure that we can use validation
    // (aren't swap chainsa n extension we care about?)
    std::vector<const char*> getRequiredExtensions ()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions (&glfwExtensionCount);

        std::vector<const char*> extensions (glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers)
        {
            extensions.push_back (VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback (
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
            throw std::runtime_error (pCallbackData->pMessage);

        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }

    // create a debug messenger using the extension function
    static VkResult CreateDebugUtilsMessengerEXT (VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
    {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr (instance, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr)
        {
            return func (instance, pCreateInfo, pAllocator, pDebugMessenger);
        }
        else
        {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }

    // destroy a debug messenger using the extension function
    void DestroyDebugUtilsMessengerEXT (VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
    {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr (instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr)
        {
            func (instance, debugMessenger, pAllocator);
        }
    }

    // initialize our test debug messenger
    VkDebugUtilsMessengerEXT debugMessenger;
    void setupDebugMessenger ()
    {
        if (!enableValidationLayers)
            return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr; // Optional

        if (CreateDebugUtilsMessengerEXT (instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error ("failed to set up debug messenger!");
        }
    }

    // Window parameters
    const int WIDTH = 800;
    const int HEIGHT = 600;

    // use GLFW to create a window
    GLFWwindow* window;
    void initWindow ()
    {
        glfwInit ();
        glfwWindowHint (GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow (WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer (window, this);

        glfwSetFramebufferSizeCallback (window, framebufferResizeCallback);
    }

    // called any time GLFW notices the window has been resized
    bool framebufferResized = false;
    static void framebufferResizeCallback (GLFWwindow* window, int width, int height)
    {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer (window));
        app->framebufferResized = true;
    }

    // create our Vulkan runtime instance
    VkInstance instance;
    void createinstance ()
    {
        if (enableValidationLayers && !checkValidationLayerSupport ())
            throw std::runtime_error ("validation layers requested, but not available!");

        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION (1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION (1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions ();
        createInfo.enabledExtensionCount = (uint32_t) extensions.size ();
        createInfo.ppEnabledExtensionNames = extensions.data ();

        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size ());
            createInfo.ppEnabledLayerNames = validationLayers.data ();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }

        VkResult result = vkCreateInstance (&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS)
            throw std::runtime_error ("Couldn't create vulkan instance");

        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties (nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> vulkanExtensions (extensionCount);
        vkEnumerateInstanceExtensionProperties (nullptr, &extensionCount, vulkanExtensions.data ());
        std::cout << "available extensions:" << std::endl;
        for (const auto& extension : vulkanExtensions)
        {
            std::cout << "\t" << extension.extensionName << std::endl;
        }
    }

    // use a GLFW Vulkan API to get the window surface
    VkSurfaceKHR surface;
    void createSurface ()
    {
        if (glfwCreateWindowSurface (instance, window, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error ("failed to create window surface");
    }

    // pick a physical device for Vulkan
    VkPhysicalDevice physicalDevice;
    void pickPhysicalDevice ()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices (instance, &deviceCount, nullptr);

        if (deviceCount == 0)
            throw std::runtime_error ("failed to find GPUs with Vulkan support!");

        uint32_t physicalDeviceCount;
        vkEnumeratePhysicalDevices (instance, &physicalDeviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices (physicalDeviceCount);
        vkEnumeratePhysicalDevices (instance, &physicalDeviceCount, devices.data ());

        for (const auto& device : devices)
        {
            if (isDeviceSuitable (device))
            {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error ("failed to find a suitable GPU!");
        }
    }

    // we need swapchain support
    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    // verify if a device suits our needs (does it have swapchain support?)
    bool isDeviceSuitable (VkPhysicalDevice device)
    {
        QueueFamilyIndices indices = findQueueFamilies (device);

        bool deviceExtensionsSupported = checkDeviceExtensionSupport (device);
        bool isSwapChainAdequate = false;
        if (deviceExtensionsSupported)
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport (device);
            isSwapChainAdequate = !(swapChainSupport.formats.empty () || swapChainSupport.presentModes.empty ());
        }

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures (device, &supportedFeatures);

        return indices.isComplete () && deviceExtensionsSupported && isSwapChainAdequate && supportedFeatures.samplerAnisotropy;
    }

    // utlity function to check for our required physical device extension
    bool checkDeviceExtensionSupport (VkPhysicalDevice device)
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties (device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions (extensionCount);
        vkEnumerateDeviceExtensionProperties (device, nullptr, &extensionCount, availableExtensions.data ());

        std::set<std::string> requiredExtensions (deviceExtensions.begin (), deviceExtensions.end ());

        for (const auto& extension : availableExtensions)
        {
            requiredExtensions.erase (extension.extensionName);
            if (requiredExtensions.empty ())
                return true;
        }

        return false;
    }

    // create the Vulkan logical device using our physical device
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkDevice logicalDevice;
    void createLogicalDevice ()
    {
        // check queue support and make our queue
        QueueFamilyIndices indices = findQueueFamilies (physicalDevice);
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies{ indices.graphicsFamily.value (), indices.presentFamily.value () };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value ();
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back (queueCreateInfo);
        }

        // create our device - enable anisotropy for texturing later on
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = (uint32_t) queueCreateInfos.size ();
        createInfo.pQueueCreateInfos = queueCreateInfos.data ();

        VkPhysicalDeviceFeatures deviceFeatures = {};
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = (uint32_t) deviceExtensions.size ();
        createInfo.ppEnabledExtensionNames = deviceExtensions.data ();

        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size ());
            createInfo.ppEnabledLayerNames = validationLayers.data ();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice (physicalDevice, &createInfo, nullptr, &logicalDevice) != VK_SUCCESS)
            throw std::runtime_error ("failed to create logical device!");

        // use our queue family indices to get handles to the graphics and presentation queues
        vkGetDeviceQueue (logicalDevice, indices.graphicsFamily.value (), 0, &graphicsQueue);
        vkGetDeviceQueue (logicalDevice, indices.presentFamily.value (), 0, &presentQueue);
    }

    // create our swapchain per our physical and logical device specifications
    // optionally supply an old swapchain - see recreateSwapChain for details
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkSwapchainKHR createSwapChain (VkSwapchainKHR oldSwapChain = VK_NULL_HANDLE)
    {
        VkSwapchainKHR ret;

        SwapChainSupportDetails swapChainSupport = querySwapChainSupport (physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat (swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode (swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent (&swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0)
            imageCount = std::min (imageCount, swapChainSupport.capabilities.maxImageCount);
        
        // were using this chain to put images (colors) on the screen
        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies (physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value (), indices.presentFamily.value () };

        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        createInfo.presentMode = presentMode;

        createInfo.clipped = VK_TRUE;

        createInfo.oldSwapchain = oldSwapChain;

        if (vkCreateSwapchainKHR (logicalDevice, &createInfo, nullptr, &ret) != VK_SUCCESS)
            throw std::runtime_error ("failed to create swap chain!");

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

        vkGetSwapchainImagesKHR (logicalDevice, ret, &imageCount, nullptr);
        swapChainImages.resize (imageCount);
        vkGetSwapchainImagesKHR (logicalDevice, ret, &imageCount, swapChainImages.data ());

        return ret;
    }

    // Create image views for each element of our swap chain
    std::vector<VkImageView> swapChainImageViews;
    void createImageViews ()
    {
        for (size_t i = 0; i < swapChainImages.size (); i++)
            swapChainImageViews.push_back (createImageView (swapChainImages[i], swapChainImageFormat));
    }

    // construct a single render pass to generate our image
    VkRenderPass renderPass;
    void createRenderPass ()
    {
        // we only care about a color pass
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        if (vkCreateRenderPass (logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
            throw std::runtime_error ("failed to create render pass");

        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;

        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;

        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
    }

    // create a descriptor set for our pipeline
    // this is used to describe our two uniforms - 
    // our MVP matrix and our image sampler
    VkDescriptorSetLayout descriptorSetLayout;
    void createDescriptorSetLayout ()
    {
        VkDescriptorSetLayoutBinding camLayoutBinding = {};
        camLayoutBinding.binding = 0;
        camLayoutBinding.descriptorCount = 1;
        camLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        camLayoutBinding.pImmutableSamplers = nullptr;
        camLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding bindings[] = { camLayoutBinding, samplerLayoutBinding };

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;


        if (vkCreateDescriptorSetLayout (logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error ("failed to create descriptor set layout!");
    }

    // create our graphics pipeline
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    void createGraphicsPipeline ()
    {
        // construct the vertex and fragment shader stages
        // (shaders must be precompiled - Vulkan has a libary to do this in code)
        VkShaderModule vertShader = createShaderModule (readFile (VERT_SHADER_NAME));
        VkShaderModule fragShader = createShaderModule (readFile (FRAG_SHADER_NAME));

        VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShader;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShader;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof (Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        // three attributes - positionm, texCoord, color
        VkVertexInputAttributeDescription attributeDescriptions[3];
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof (Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof (Vertex, tex);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[2].offset = offsetof (Vertex, color);

        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = 3;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        // rasterization (so we can draw our square)
        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        // disabled multisampling (TODO)
        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // color blend specifications - alpha blend [0, 1]
        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout (logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error ("failed to create pipeline layout!");

        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;

        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines (logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
            throw std::runtime_error ("failed to create graphics pipeline!");

        vkDestroyShaderModule (logicalDevice, vertShader, nullptr);
        vkDestroyShaderModule (logicalDevice, fragShader, nullptr);
    }

    // create frame buffers for our swap chain image views
    std::vector<VkFramebuffer> swapChainFrameBuffers;
    void createFrameBuffers ()
    {
        swapChainFrameBuffers.resize (swapChainImageViews.size ());
        for (size_t i = 0; i < swapChainImageViews.size (); i++)
        {
            VkImageView attachments[] = { swapChainImageViews[i] };

            VkFramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer (logicalDevice, &framebufferInfo, nullptr, &swapChainFrameBuffers[i]) != VK_SUCCESS)
                throw std::runtime_error ("failed to create framebuffer!");
        }
    }

    // construc the Vulkan command pool
    // this gets shared by all of our command buffers - we could make separate pool...
    VkCommandPool commandPool;
    void createCommandPool ()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies (physicalDevice);

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value ();
        poolInfo.flags = 0; // Optional
        if (vkCreateCommandPool (logicalDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
            throw std::runtime_error ("failed to create command pool");
    }

    // Create a texture image
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    void createTextureImage ()
    {
        // use STB to load an image file
        int texWidth, texHeight, texChannels;
        std::string texName = TEX_IMG_NAME;
        stbi_uc* pixels = stbi_load (texName.c_str (), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (pixels == nullptr)
            throw std::runtime_error ("failed to load texture image " + texName);

        // create our staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer (imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, &stagingBufferMemory);

        // map our staging buffer memory and copy the image
        // (can Vulkan map the STB image's memory directly?)
        void* pixelData;
        vkMapMemory (logicalDevice, stagingBufferMemory, 0, imageSize, 0, &pixelData);
        memcpy (pixelData, pixels, (size_t) imageSize);
        vkUnmapMemory (logicalDevice, stagingBufferMemory);

        stbi_image_free (pixels);

        // create image, transfer to device texture image, transition again for shader reads
        createImage (texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &textureImage, &textureImageMemory);

        transitionImageLayout (textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage (stagingBuffer, textureImage, (uint32_t) texWidth, (uint32_t) texHeight);

        transitionImageLayout (textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer (logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory (logicalDevice, stagingBufferMemory, nullptr);
    }

    // create an image view for our texture image so it can be used in the pipeline
    VkImageView textureImageView;
    void createTextureImageView ()
    {
        textureImageView = createImageView (textureImage, VK_FORMAT_R8G8B8A8_UNORM);
    }

    // create a texture sampler
    VkSampler textureSampler;
    void createTextureSampler ()
    {
        VkSamplerCreateInfo samplerInfo = {};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;

        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16;

        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        if (vkCreateSampler (logicalDevice, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
            throw std::runtime_error ("failed to create texture sampler!");
    }

    // create our vertex buffer, containing per vertex attribute data
    const std::vector<Vertex> vertices = {
        // pos            tex           color 
        {{-0.5f, -0.5f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, -0.5f},  {0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
        {{0.5f, 0.5f},   {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f},  {1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}
    };
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    void createVertexBuffer ()
    {
        VkDeviceSize vertexBufferSize = sizeof (vertices[0]) * vertices.size ();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        createBuffer (vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, &stagingMemory);

        void* vertexData;
        vkMapMemory (logicalDevice, stagingMemory, 0, vertexBufferSize, 0, &vertexData);
        memcpy (vertexData, vertices.data (), (size_t) vertexBufferSize);
        vkUnmapMemory (logicalDevice, stagingMemory);

        createBuffer (vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &vertexBuffer, &vertexBufferMemory);

        copyBuffer (stagingBuffer, vertexBuffer, vertexBufferSize);

        vkDestroyBuffer (logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory (logicalDevice, stagingMemory, nullptr);
    }

    // construct our index buffer for drawing our square
    const std::vector<uint16_t> indices = {
        0, 1, 2, 2, 3, 0
    };
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    void createIndexBuffer ()
    {
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        VkDeviceSize indexBufferSize = sizeof (uint16_t) * indices.size ();
        createBuffer (indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, &stagingMemory);

        void* indexData;
        vkMapMemory (logicalDevice, stagingMemory, 0, indexBufferSize, 0, &indexData);
        memcpy (indexData, indices.data (), (size_t) indexBufferSize);
        vkUnmapMemory (logicalDevice, stagingMemory);

        createBuffer (indexBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &indexBuffer, &indexBufferMemory);

        copyBuffer (stagingBuffer, indexBuffer, indexBufferSize);

        vkDestroyBuffer (logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory (logicalDevice, stagingMemory, nullptr);
    }

    // create a uniform buffer memory block for each swap chain image
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBufferMemory;
    void createUniformBuffers ()
    {
        VkDeviceSize bufferSize = sizeof (Camera);

        uniformBuffers.resize (swapChainImages.size ());
        uniformBufferMemory.resize (swapChainImages.size ());

        for (size_t i = 0; i < swapChainImages.size (); i++)
            createBuffer (bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &uniformBuffers[i], &uniformBufferMemory[i]);
    }
    
    // create a descriptor pool for our MVP matrix and texture sampler
    VkDescriptorPool descriptorPool;
    void createDescriptorPool ()
    {
        VkDescriptorPoolSize poolSizes[2] = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = (uint32_t) swapChainImages.size ();
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = (uint32_t) swapChainImages.size ();

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        poolInfo.maxSets = (uint32_t) swapChainImages.size ();

        if (vkCreateDescriptorPool (logicalDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
            throw std::runtime_error ("failed to create descriptor pool!");
    }

    // create a descriptor set for each of our images
    std::vector<VkDescriptorSet> descriptorSets;
    void createDescriptorSets ()
    {
        std::vector<VkDescriptorSetLayout> layouts (swapChainImages.size (), descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = (uint32_t) swapChainImages.size ();
        allocInfo.pSetLayouts = layouts.data ();

        descriptorSets.resize (swapChainImages.size ());
        if (vkAllocateDescriptorSets (logicalDevice, &allocInfo, descriptorSets.data ()) != VK_SUCCESS)
            throw std::runtime_error ("failed to allocate descriptor sets!");

        // we need two descriptor sets for our two uniforms
        for (size_t i = 0; i < swapChainImages.size (); i++)
        {
            VkWriteDescriptorSet descriptorWriteSets[2] = {};

            // prepare to write matrix
            VkDescriptorBufferInfo bufferInfo = {};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof (Camera);

            descriptorWriteSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWriteSets[0].dstSet = descriptorSets[i];
            descriptorWriteSets[0].dstBinding = 0;
            descriptorWriteSets[0].dstArrayElement = 0;
            descriptorWriteSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWriteSets[0].descriptorCount = 1;
            descriptorWriteSets[0].pBufferInfo = &bufferInfo;

            // prepare to wrtie sampler
            VkDescriptorImageInfo imageInfo = {};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            descriptorWriteSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWriteSets[1].dstSet = descriptorSets[i];
            descriptorWriteSets[1].dstBinding = 1;
            descriptorWriteSets[1].dstArrayElement = 0;
            descriptorWriteSets[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWriteSets[1].descriptorCount = 1;
            descriptorWriteSets[1].pImageInfo = &imageInfo;

            vkUpdateDescriptorSets (logicalDevice, 2, descriptorWriteSets, 0, nullptr);
        }
    }

    // create our per-image command buffers
    std::vector<VkCommandBuffer> commandBuffers;
    void createCommandBuffers ()
    {
        commandBuffers.resize (swapChainFrameBuffers.size ());
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size ();

        if (vkAllocateCommandBuffers (logicalDevice, &allocInfo, commandBuffers.data ()) != VK_SUCCESS)
            throw std::runtime_error ("failed to allocate command buffers!");

        // record each image's command buffer
        for (size_t i = 0; i < commandBuffers.size (); i++)
        {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            beginInfo.pInheritanceInfo = nullptr; // Optional

            if (vkBeginCommandBuffer (commandBuffers[i], &beginInfo) != VK_SUCCESS)
                throw std::runtime_error ("failed to begin recording command buffer!");

            VkRenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFrameBuffers[i];
            renderPassInfo.renderArea.offset = { 0,0 };
            renderPassInfo.renderArea.extent = swapChainExtent;

            VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;

            // begin a render pass
            // clear the color to black
            vkCmdBeginRenderPass (commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            // bind our graphics pipeline
            vkCmdBindPipeline (commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            // bind our vertex buffers
            VkBuffer vertexBuffers[] = { vertexBuffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers (commandBuffers[i], 0, 1, vertexBuffers, offsets);

            // bind our index buffer
            vkCmdBindIndexBuffer (commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);

            // bind our uniform buffers
            vkCmdBindDescriptorSets (commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

            // draw
            vkCmdDrawIndexed (commandBuffers[i], (uint32_t) indices.size (), 1, 0, 0, 0);

            // end render pass
            vkCmdEndRenderPass (commandBuffers[i]);

            if (vkEndCommandBuffer (commandBuffers[i]) != VK_SUCCESS)
                throw std::runtime_error ("failed to record command buffer!");
        }
    }
    
    // use these objects to synchronize host command with device interactions
    const int MAX_FRAMES_IN_FLIGHT = 2;

    struct RenderSync
    {
        // semaphores for when a host image is available 
        // and a device render has finished
        VkSemaphore imageAvailable, renderFinished;
        
        // use this fence to prevent more than MAX_FRAMES_IN_FLIGHT from building up
        VkFence inFlightFence;
    };
    std::vector<RenderSync> renderSyncs;
    void createSyncObjects ()
    {
        renderSyncs.resize (MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (auto& renderSync : renderSyncs)
        {
            for (VkSemaphore* semaphore : { &renderSync.imageAvailable, &renderSync.renderFinished })
                if (vkCreateSemaphore (logicalDevice, &semaphoreInfo, nullptr, semaphore) != VK_SUCCESS)
                    throw std::runtime_error ("failed to create semaphore!");

            if (vkCreateFence (logicalDevice, &fenceInfo, nullptr, &renderSync.inFlightFence) != VK_SUCCESS)
                throw std::runtime_error ("failed to create fence!");
        }
    }

    // init vulkan - see above functions
    void initVulkan ()
    {
        createinstance ();
        setupDebugMessenger ();
        createSurface ();
        pickPhysicalDevice ();
        createLogicalDevice ();

        swapChain = createSwapChain ();
        createImageViews ();
        createRenderPass ();
        createDescriptorSetLayout ();
        createGraphicsPipeline ();
        createFrameBuffers ();
        createCommandPool ();
        createTextureImage ();
        createTextureImageView ();
        createTextureSampler ();
        createVertexBuffer ();
        createIndexBuffer ();
        createUniformBuffers ();
        createDescriptorPool ();
        createDescriptorSets ();
        createCommandBuffers ();
        createSyncObjects ();
    }

    size_t currentFrame = 0;
    void drawFrame ()
    {
        // get render sync, wait for inflight fences to let us through
        auto& renderSync = renderSyncs[currentFrame];
        vkWaitForFences (logicalDevice, 1, &renderSync.inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max ());

        // get the next image, possibly recreate the swap chain if the frame is out of date
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR (logicalDevice, swapChain, std::numeric_limits<uint64_t>::max (), renderSync.imageAvailable, VK_NULL_HANDLE, &imageIndex);
        switch (result)
        {
        case VK_ERROR_OUT_OF_DATE_KHR:
            recreateSwapChain ();
            return;
        case VK_SUCCESS:
        case VK_SUBOPTIMAL_KHR:
            break;
        default:
            throw std::runtime_error ("failed to acquire swapchain image!");
        }

        // update our uniforms (rotate MVP)
        updateUniformBuffers (imageIndex);

        // prepare to submit our render pass command buffer
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { renderSync.imageAvailable };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        VkSemaphore signalSemaphores[] = { renderSync.renderFinished };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        vkResetFences (logicalDevice, 1, &renderSync.inFlightFence);

        if (vkQueueSubmit (graphicsQueue, 1, &submitInfo, renderSync.inFlightFence) != VK_SUCCESS)
            throw std::runtime_error ("failed to submit draw command buffer!");

        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        // present a complete image
        result = vkQueuePresentKHR (presentQueue, &presentInfo);

        // maybe handle a resize here
        if (framebufferResized)
            result = VK_ERROR_OUT_OF_DATE_KHR;

        switch (result)
        {
        case VK_ERROR_OUT_OF_DATE_KHR:
        case VK_SUBOPTIMAL_KHR:
            recreateSwapChain ();
            framebufferResized = false;
        case VK_SUCCESS:
            break;
        default:
            throw std::runtime_error ("failed to acquire swapchain image!");
        }

        // advance frame count
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // rotate M matrix over time
    void updateUniformBuffers (uint32_t currentImage)
    {
        static auto startTime = std::chrono::high_resolution_clock::now ();

        auto currentTime = std::chrono::high_resolution_clock::now ();
        float time = std::chrono::duration<float, std::chrono::seconds::period> (currentTime - startTime).count ();

        Camera cam;
        cam.M = glm::rotate (glm::mat4 (1.0f), time * glm::radians (90.0f), glm::vec3 (0.0f, 0.0f, 1.0f));
        cam.V = glm::lookAt (glm::vec3 (2.0f, 2.0f, 2.0f), glm::vec3 (0.0f, 0.0f, 0.0f), glm::vec3 (0.0f, 0.0f, 1.0f));
        cam.P = glm::perspective (glm::radians (45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
        cam.P[1][1] *= -1;

        void* data;
        vkMapMemory (logicalDevice, uniformBufferMemory[currentImage], 0, sizeof (cam), 0, &data);
        memcpy (data, &cam, sizeof (cam));
        vkUnmapMemory (logicalDevice, uniformBufferMemory[currentImage]);
    }


    void mainLoop ()
    {
        while (!glfwWindowShouldClose (window))
        {
            glfwPollEvents ();
            drawFrame ();
        }

        vkDeviceWaitIdle (logicalDevice);
    }

    void cleanUpSwapChain ()
    {
        for (auto& frameBuffer : swapChainFrameBuffers)
            vkDestroyFramebuffer (logicalDevice, frameBuffer, nullptr);

        vkFreeCommandBuffers (logicalDevice, commandPool, (uint32_t) commandBuffers.size (), commandBuffers.data ());

        vkDestroyPipeline (logicalDevice, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout (logicalDevice, pipelineLayout, nullptr);
        vkDestroyRenderPass (logicalDevice, renderPass, nullptr);

        for (auto& imageView : swapChainImageViews)
            vkDestroyImageView (logicalDevice, imageView, nullptr);

        vkDestroySwapchainKHR (logicalDevice, swapChain, nullptr);
    }

    void recreateSwapChain ()
    {
        // in case we're minimized
        int width = 0, height = 0;
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize (window, &width, &height);
            glfwWaitEvents ();
        }

        VkSwapchainKHR newSwapChain = createSwapChain (swapChain);
        vkDeviceWaitIdle (logicalDevice);
        cleanUpSwapChain ();
        swapChain = newSwapChain;

        createImageViews ();
        createRenderPass ();
        createGraphicsPipeline ();
        createFrameBuffers ();
        createCommandBuffers ();
    }

    void cleanup ()
    {
        cleanUpSwapChain ();

        vkDestroySampler (logicalDevice, textureSampler, nullptr);
        vkDestroyImageView (logicalDevice, textureImageView, nullptr);

        vkDestroyImage (logicalDevice, textureImage, nullptr);
        vkFreeMemory (logicalDevice, textureImageMemory, nullptr);

        vkDestroyDescriptorPool (logicalDevice, descriptorPool, nullptr);

        vkDestroyDescriptorSetLayout (logicalDevice, descriptorSetLayout, nullptr);

        for (size_t i = 0; i < swapChainImages.size (); i++)
        {
            vkDestroyBuffer (logicalDevice, uniformBuffers[i], nullptr);
            vkFreeMemory (logicalDevice, uniformBufferMemory[i], nullptr);
        }

        vkDestroyBuffer (logicalDevice, vertexBuffer, nullptr);
        vkFreeMemory (logicalDevice, vertexBufferMemory, nullptr);

        vkDestroyBuffer (logicalDevice, indexBuffer, nullptr);
        vkFreeMemory (logicalDevice, indexBufferMemory, nullptr);

        for (auto& renderSync : renderSyncs)
        {
            for (VkSemaphore* semaphore : { &renderSync.imageAvailable, &renderSync.renderFinished })
                vkDestroySemaphore (logicalDevice, *semaphore, nullptr);
            vkDestroyFence (logicalDevice, renderSync.inFlightFence, nullptr);
        }

        vkDestroyCommandPool (logicalDevice, commandPool, nullptr);

        vkDestroyDevice (logicalDevice, nullptr);

        if (enableValidationLayers)
            DestroyDebugUtilsMessengerEXT (instance, debugMessenger, nullptr);

        vkDestroySurfaceKHR (instance, surface, nullptr);

        vkDestroyInstance (instance, nullptr);

        glfwDestroyWindow (window);

        glfwTerminate ();
    }

    bool checkValidationLayerSupport ()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties (&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers (layerCount);
        vkEnumerateInstanceLayerProperties (&layerCount, availableLayers.data ());
        for (auto& layerName : validationLayers)
            if (std::none_of (availableLayers.begin (), availableLayers.end (),
                [layerName](VkLayerProperties props) { return strcmp (layerName, props.layerName) == 0; }))
                return false;
        return true;
    }
};

int main ()
{
    HelloTriangleApplication app;

    try
    {
        app.run ();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what () << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}